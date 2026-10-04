import copy

import pytest
import torch

from model.variable.parameter import Bias, DenseWeight, PoolWeight
from training.core.optimizers import SGDOptimizer, build_optimizer
from training.tiki_taka import TikiTakaConfig, TikiTakaOptimizer, parse_update_pipeline


class _ParameterFunction:
    """Minimal function object exposing the optimizer's required interface."""

    def __init__(self, *parameters):
        self._parameters = list(parameters)

    def params(self):
        return self._parameters


def _dense(input_size=1, output_size=1, *, clamp=False, clamp_min=None, clamp_max=None):
    parameter = DenseWeight(
        layer_pre_shape=(input_size,),
        layer_post_shape=(output_size,),
        gain=0.0,
        device="cpu",
        clamp=clamp,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )
    parameter.state.zero_()
    return parameter


def _bias(size=1):
    parameter = Bias(shape=(size,), gain=0.0, device="cpu")
    parameter.state.zero_()
    return parameter


def _optimizer(parameter, *, learning_rate=1.0, **config):
    return TikiTakaOptimizer(
        _ParameterFunction(parameter),
        _ParameterFunction(),
        [learning_rate],
        TikiTakaConfig(**config),
    )


def _set_gradient(parameter, values):
    parameter.state.grad = torch.as_tensor(
        values,
        dtype=parameter.state.dtype,
        device=parameter.state.device,
    ).reshape_as(parameter.state)


@pytest.mark.parametrize("update_pipeline", [None, {"type": "direct"}])
def test_build_optimizer_preserves_direct_sgd_factory_behavior(update_pipeline):
    parameter = _dense(output_size=2)
    optimizer = build_optimizer(
        _ParameterFunction(parameter),
        _ParameterFunction(),
        [0.25],
        update_pipeline=update_pipeline,
        momentum=0.0,
        weight_decay=0.0,
    )

    assert isinstance(optimizer, SGDOptimizer)
    assert not isinstance(optimizer, TikiTakaOptimizer)

    _set_gradient(parameter, [2.0, -4.0])
    optimizer.step()

    torch.testing.assert_close(parameter.state, torch.tensor([[-0.5, 1.0]]))


def test_gradients_accumulate_without_changing_visible_weights_before_transfer():
    parameter = _dense(output_size=2)
    optimizer = _optimizer(
        parameter,
        fast_lr=0.5,
        transfer_every=3,
        n_reads_per_transfer=1,
    )

    for _ in range(2):
        _set_gradient(parameter, [2.0, -4.0])
        optimizer.step()

    torch.testing.assert_close(parameter.state, torch.zeros_like(parameter.state))
    torch.testing.assert_close(
        optimizer.auxiliary_state(parameter.state),
        torch.tensor([[-2.0, 4.0]]),
    )
    assert optimizer.state[parameter.state]["step"] == 2
    assert optimizer.state[parameter.state]["transfer_count"] == 0


def test_transfer_uses_negative_gradient_and_scaled_visible_learning_rate():
    parameter = _dense(output_size=2)
    optimizer = _optimizer(
        parameter,
        learning_rate=0.25,
        fast_lr=2.0,
        transfer_every=1,
        transfer_lr=3.0,
        scale_transfer_lr=True,
    )
    parameter.state.copy_(torch.tensor([[10.0, 20.0]]))

    _set_gradient(parameter, [2.0, -4.0])
    optimizer.step()

    torch.testing.assert_close(
        optimizer.auxiliary_state(parameter.state),
        torch.tensor([[-4.0, 8.0]]),
    )
    # Visible += auxiliary * transfer_lr * per-parameter learning_rate.
    torch.testing.assert_close(parameter.state, torch.tensor([[7.0, 26.0]]))


def test_dense_logical_column_transfers_advance_and_wrap_cyclically():
    parameter = _dense(input_size=2, output_size=3)
    optimizer = _optimizer(
        parameter,
        transfer_every=1,
        n_reads_per_transfer=1,
        transfer_lr=1.0,
        scale_transfer_lr=False,
    )
    gradient = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    _set_gradient(parameter, gradient)
    optimizer.step()
    torch.testing.assert_close(
        parameter.state,
        torch.tensor([[-1.0, -2.0, -3.0], [0.0, 0.0, 0.0]]),
    )
    assert optimizer.state[parameter.state]["transfer_index"] == 1

    _set_gradient(parameter, gradient)
    optimizer.step()
    torch.testing.assert_close(
        parameter.state,
        torch.tensor([[-1.0, -2.0, -3.0], [-8.0, -10.0, -12.0]]),
    )
    assert optimizer.state[parameter.state]["transfer_index"] == 0

    _set_gradient(parameter, gradient)
    optimizer.step()
    torch.testing.assert_close(
        parameter.state,
        torch.tensor([[-4.0, -8.0, -12.0], [-8.0, -10.0, -12.0]]),
    )
    assert optimizer.state[parameter.state]["transfer_index"] == 1
    assert optimizer.state[parameter.state]["transfer_count"] == 3


def test_dense_logical_row_transfers_select_output_rows():
    parameter = _dense(input_size=2, output_size=3)
    optimizer = _optimizer(
        parameter,
        transfer_every=1,
        n_reads_per_transfer=1,
        transfer_lr=1.0,
        scale_transfer_lr=False,
        transfer_columns=False,
    )
    gradient = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    _set_gradient(parameter, gradient)
    optimizer.step()
    torch.testing.assert_close(
        parameter.state,
        torch.tensor([[-1.0, 0.0, 0.0], [-4.0, 0.0, 0.0]]),
    )

    _set_gradient(parameter, gradient)
    optimizer.step()
    torch.testing.assert_close(
        parameter.state,
        torch.tensor([[-1.0, -4.0, 0.0], [-4.0, -10.0, 0.0]]),
    )
    assert optimizer.state[parameter.state]["transfer_index"] == 2


def test_fast_lr_zero_uses_the_current_per_parameter_optimizer_rate():
    parameter = _dense(output_size=2)
    optimizer = _optimizer(
        parameter,
        learning_rate=0.25,
        fast_lr=0.0,
        transfer_every=0,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.4)

    _set_gradient(parameter, [2.0, -4.0])
    optimizer.step()
    scheduler.step()
    torch.testing.assert_close(
        optimizer.auxiliary_state(parameter.state),
        torch.tensor([[-0.5, 1.0]]),
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)

    _set_gradient(parameter, [2.0, -4.0])
    optimizer.step()
    torch.testing.assert_close(
        optimizer.auxiliary_state(parameter.state),
        torch.tensor([[-0.7, 1.4]]),
    )


@pytest.mark.parametrize(
    ("reset_probability", "expected_auxiliary"),
    [
        (0.0, [[-1.0, -1.0], [-1.0, -1.0]]),
        (1.0, [[0.0, 0.0], [-1.0, -1.0]]),
    ],
)
def test_transfer_retains_or_resets_only_the_selected_slice(
    reset_probability,
    expected_auxiliary,
):
    parameter = _dense(input_size=2, output_size=2)
    optimizer = _optimizer(
        parameter,
        transfer_every=1,
        n_reads_per_transfer=1,
        transfer_lr=1.0,
        scale_transfer_lr=False,
        with_reset_prob=reset_probability,
    )

    _set_gradient(parameter, torch.ones_like(parameter.state))
    optimizer.step()

    torch.testing.assert_close(
        parameter.state,
        torch.tensor([[-1.0, -1.0], [0.0, 0.0]]),
    )
    torch.testing.assert_close(
        optimizer.auxiliary_state(parameter.state),
        torch.tensor(expected_auxiliary),
    )


def test_auxiliary_is_signed_even_when_visible_parameter_is_clamped():
    parameter = _dense(
        output_size=2,
        clamp=True,
        clamp_min=0.0,
        clamp_max=1.0,
    )
    optimizer = _optimizer(
        parameter,
        transfer_every=1,
        transfer_lr=1.0,
        scale_transfer_lr=False,
    )

    _set_gradient(parameter, [2.0, -0.5])
    optimizer.step()
    torch.testing.assert_close(parameter.state, torch.tensor([[-2.0, 0.5]]))

    # Trainers clamp visible DRN parameters after optimizer.step(). The hidden
    # array must remain signed and independent of that physical weight bound.
    parameter.clamp_()
    torch.testing.assert_close(parameter.state, torch.tensor([[0.0, 0.5]]))
    torch.testing.assert_close(
        optimizer.auxiliary_state(parameter.state),
        torch.tensor([[-2.0, 0.5]]),
    )


def test_zero_grad_clears_gradient_but_preserves_auxiliary_and_transfer_state():
    parameter = _dense(output_size=2)
    optimizer = _optimizer(parameter, transfer_every=0, fast_lr=0.5)

    _set_gradient(parameter, [2.0, -4.0])
    optimizer.step()
    auxiliary_before = optimizer.auxiliary_state(parameter.state).clone()
    state_before = {
        key: optimizer.state[parameter.state][key]
        for key in ("step", "transfer_index", "transfer_count")
    }

    optimizer.zero_grad()

    assert parameter.state.grad is None
    torch.testing.assert_close(
        optimizer.auxiliary_state(parameter.state),
        auxiliary_before,
    )
    assert {
        key: optimizer.state[parameter.state][key]
        for key in ("step", "transfer_index", "transfer_count")
    } == state_before

    _set_gradient(parameter, [2.0, -4.0])
    optimizer.step()
    torch.testing.assert_close(
        optimizer.auxiliary_state(parameter.state),
        2.0 * auxiliary_before,
    )


def test_per_parameter_schedule_and_default_direct_bias_update():
    first_weight = _dense(output_size=2)
    second_weight = _dense(output_size=2)
    bias = _bias(size=2)
    pool = PoolWeight(
        shape=(1, 1, 1, 1),
        gain=1.0,
        device="cpu",
        clamp=False,
    )
    energy_fn = _ParameterFunction(first_weight, pool, second_weight, bias)
    optimizer = TikiTakaOptimizer(
        energy_fn,
        _ParameterFunction(),
        [0.5, 0.25, 0.1],
        TikiTakaConfig(transfer_every=2),
    )

    _set_gradient(first_weight, [1.0, 1.0])
    _set_gradient(bias, [2.0, -4.0])
    optimizer.step()

    torch.testing.assert_close(first_weight.state, torch.zeros_like(first_weight.state))
    torch.testing.assert_close(second_weight.state, torch.zeros_like(second_weight.state))
    torch.testing.assert_close(bias.state, torch.tensor([-0.2, 0.4]))
    assert optimizer.state[first_weight.state]["step"] == 1
    assert optimizer.auxiliary_state(second_weight.state) is None
    assert all(
        pool.state is not tensor
        for group in optimizer.param_groups
        for tensor in group["params"]
    )

    _set_gradient(first_weight, [1.0, 1.0])
    _set_gradient(second_weight, [1.0, 1.0])
    bias.state.grad = None
    optimizer.step()

    # The first weight reaches its second local write and transfers. The second
    # weight has only seen one gradient, so its visible value remains unchanged.
    torch.testing.assert_close(first_weight.state, torch.tensor([[-1.0, -1.0]]))
    torch.testing.assert_close(second_weight.state, torch.zeros_like(second_weight.state))
    assert optimizer.state[first_weight.state]["step"] == 2
    assert optimizer.state[second_weight.state]["step"] == 1


def test_bias_can_opt_into_auxiliary_accumulation():
    bias = _bias(size=2)
    optimizer = _optimizer(
        bias,
        learning_rate=0.25,
        transfer_every=2,
        transfer_lr=1.0,
        scale_transfer_lr=True,
        accumulate_biases=True,
    )

    _set_gradient(bias, [2.0, -4.0])
    optimizer.step()
    torch.testing.assert_close(bias.state, torch.zeros_like(bias.state))

    _set_gradient(bias, [2.0, -4.0])
    optimizer.step()

    torch.testing.assert_close(bias.state, torch.tensor([-1.0, 2.0]))
    torch.testing.assert_close(
        optimizer.auxiliary_state(bias.state),
        torch.tensor([-4.0, 8.0]),
    )


def test_parse_update_pipeline_accepts_supported_forms():
    config = TikiTakaConfig(transfer_every=7)

    assert parse_update_pipeline(None) is None
    assert parse_update_pipeline({"type": "direct"}) is None
    assert parse_update_pipeline(config) is config
    assert parse_update_pipeline(
        {
            "type": "tiki_taka",
            "fast_lr": 0.25,
            "transfer_every": 7,
            "transfer_columns": False,
        }
    ) == TikiTakaConfig(
        fast_lr=0.25,
        transfer_every=7,
        transfer_columns=False,
    )
    assert parse_update_pipeline(
        {
            "type": "tiki_taka",
            "aihwkit_preset": "TikiTakaReRamESPreset",
            "aihwkit_conductance_min": 1e-7,
            "aihwkit_conductance_max": 1.0,
            "aihwkit_construction_seed": 7,
        }
    ) == TikiTakaConfig(
        aihwkit_preset="TikiTakaReRamESPreset",
        aihwkit_conductance_min=1e-7,
        aihwkit_conductance_max=1.0,
        aihwkit_construction_seed=7,
    )


@pytest.mark.parametrize(
    "value",
    [
        42,
        {},
        {"type": "unknown"},
        {"type": "direct", "fast_lr": 1.0},
        {"type": "tiki_taka", "unknown_key": 1},
        {"type": "tiki_taka", "fast_lr": -1.0},
        {"type": "tiki_taka", "transfer_every": True},
        {"type": "tiki_taka", "n_reads_per_transfer": 0},
        {"type": "tiki_taka", "with_reset_prob": 1.1},
        {
            "type": "tiki_taka",
            "transfer_columns": False,
            "with_reset_prob": 0.5,
        },
        {"type": "tiki_taka", "scale_transfer_lr": 1},
        {
            "type": "tiki_taka",
            "fast_weight_min": 1.0,
            "fast_weight_max": 1.0,
        },
        {"type": "tiki_taka", "aihwkit_preset": "unknown"},
        {
            "type": "tiki_taka",
            "aihwkit_conductance_min": 0.0,
            "aihwkit_conductance_max": 1.0,
        },
        {
            "type": "tiki_taka",
            "aihwkit_preset": "TikiTakaIdealizedPreset",
            "aihwkit_conductance_min": -1.0,
        },
        {
            "type": "tiki_taka",
            "aihwkit_preset": "TikiTakaIdealizedPreset",
            "aihwkit_conductance_max": 0.0,
        },
        {
            "type": "tiki_taka",
            "aihwkit_preset": "TikiTakaIdealizedPreset",
            "aihwkit_conductance_min": 1.0,
            "aihwkit_conductance_max": 1.0,
        },
        {
            "type": "tiki_taka",
            "aihwkit_preset": "TikiTakaIdealizedPreset",
            "aihwkit_construction_seed": 0,
        },
        {
            "type": "tiki_taka",
            "aihwkit_preset": "TikiTakaIdealizedPreset",
            "fast_weight_min": 0.1,
        },
    ],
)
def test_parse_update_pipeline_rejects_invalid_configuration_with_context(value):
    with pytest.raises(ValueError) as exc_info:
        parse_update_pipeline(value)

    message = str(exc_info.value)
    assert message.startswith("Expected")
    assert "Provided value:" in message


def test_optimizer_rejects_unsupported_sgd_options_and_rate_count():
    parameter = _dense()
    energy_fn = _ParameterFunction(parameter)
    cost_fn = _ParameterFunction()
    config = TikiTakaConfig()

    with pytest.raises(ValueError, match="Expected momentum=0 and weight_decay=0"):
        TikiTakaOptimizer(
            energy_fn,
            cost_fn,
            [0.1],
            config,
            momentum=0.9,
        )

    with pytest.raises(ValueError, match="Expected 1 learning rates"):
        TikiTakaOptimizer(energy_fn, cost_fn, [], config)


def test_optimizer_state_dict_round_trip_preserves_pipeline_and_continuation():
    first_parameter = _dense(input_size=2, output_size=2)
    config = TikiTakaConfig(
        fast_lr=0.5,
        transfer_every=1,
        n_reads_per_transfer=1,
        transfer_lr=2.0,
        scale_transfer_lr=True,
    )
    first_optimizer = TikiTakaOptimizer(
        _ParameterFunction(first_parameter),
        _ParameterFunction(),
        [0.25],
        config,
    )
    gradient = [[1.0, -2.0], [3.0, -4.0]]
    for _ in range(2):
        _set_gradient(first_parameter, gradient)
        first_optimizer.step()

    saved_state = copy.deepcopy(first_optimizer.state_dict())
    second_parameter = _dense(input_size=2, output_size=2)
    second_parameter.state.copy_(first_parameter.state)
    second_optimizer = TikiTakaOptimizer(
        _ParameterFunction(second_parameter),
        _ParameterFunction(),
        [0.25],
        config,
    )
    second_optimizer.load_state_dict(saved_state)

    first_state = first_optimizer.state[first_parameter.state]
    second_state = second_optimizer.state[second_parameter.state]
    torch.testing.assert_close(second_state["auxiliary"], first_state["auxiliary"])
    assert second_state["step"] == first_state["step"] == 2
    assert second_state["transfer_index"] == first_state["transfer_index"] == 0
    assert second_state["transfer_count"] == first_state["transfer_count"] == 2

    for parameter, optimizer in (
        (first_parameter, first_optimizer),
        (second_parameter, second_optimizer),
    ):
        _set_gradient(parameter, [[0.5, 1.0], [-0.5, -1.0]])
        optimizer.step()

    torch.testing.assert_close(second_parameter.state, first_parameter.state)
    torch.testing.assert_close(
        second_optimizer.auxiliary_state(second_parameter.state),
        first_optimizer.auxiliary_state(first_parameter.state),
    )
    assert (
        second_optimizer.state[second_parameter.state]["transfer_index"]
        == first_optimizer.state[first_parameter.state]["transfer_index"]
    )
