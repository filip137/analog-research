import torch

from model.function.cost import SquaredError
from model.function.network import Network
from model.resistive.minimizer import QuadraticMinimizer
from model.resistive.network import DeepResistiveEnergy
from model.variable.parameter import Bias, DenseWeight
from training.core.sgd import AugmentedFunction, EquilibriumProp
from training.core.optimizers import build_optimizer
from training.tiki_taka import TikiTakaOptimizer


def _build_minimizer(fn, free_layers, diode_params):
    return QuadraticMinimizer(
        fn=fn,
        free_layers=free_layers,
        num_iterations=4,
        mode="asynchronous",
        non_linearity="perfect_diode",
        quadratic_diode_param=diode_params["quadratic"],
        exponential_diode_param=diode_params["exponential"],
        hard_sigmoid_param=diode_params["hard_sigmoid"],
        voltage_amp=1.0,
        current_amp=1.0,
    )


def test_mlp_eqprop_gradients_accumulate_then_transfer_to_visible_weights():
    """Exercise the real MLP/EqProp path through two Tiki-Taka minibatches."""

    torch.manual_seed(7)
    diode_params = {
        "quadratic": {
            "diode_conductance": 10.0,
            "v_min": -1.0,
            "v_max": 1.0,
        },
        "exponential": {
            "I_s": 1e-6,
            "V_t": 0.05,
            "V_off": 0.5,
        },
        "hard_sigmoid": {
            "g_on": 10.0,
            "g_off": 10.0,
            "v_min": -1.0,
            "v_max": 1.0,
        },
    }
    energy_fn = DeepResistiveEnergy(
        layer_shapes=[(4,), (4,), (2,)],
        weight_gains=[0.2, 0.2],
        input_gain=1.0,
        non_linearity="perfect_diode",
        quadratic_diode_param=diode_params["quadratic"],
        exponential_diode_param=diode_params["exponential"],
        hard_sigmoid_param=diode_params["hard_sigmoid"],
        voltage_amp=1.0,
        current_amp=1.0,
        weight_min=1e-7,
        weight_max=2.0,
    )
    energy_fn.set_device(torch.device("cpu"))

    network = Network(energy_fn)
    free_layers = network.free_layers()
    cost_fn = SquaredError(energy_fn.layers()[-1])
    augmented_fn = AugmentedFunction(energy_fn, cost_fn)
    free_minimizer = _build_minimizer(energy_fn, free_layers, diode_params)
    training_minimizer = _build_minimizer(augmented_fn, free_layers, diode_params)

    parameters = energy_fn.params()
    weights = [parameter for parameter in parameters if isinstance(parameter, DenseWeight)]
    assert len(weights) == 2
    estimator = EquilibriumProp(
        parameters,
        free_layers,
        augmented_fn,
        cost_fn,
        training_minimizer,
        variant="centered",
        nudging=0.1,
    )
    optimizer = build_optimizer(
        energy_fn,
        cost_fn,
        [
            0.0 if isinstance(parameter, Bias) else 0.05
            for parameter in parameters
        ],
        update_pipeline={
            "type": "tiki_taka",
            "fast_lr": 1.0,
            "transfer_every": 2,
            "n_reads_per_transfer": 64,
            "transfer_lr": 1.0,
            "scale_transfer_lr": True,
            "with_reset_prob": 1.0,
        },
    )
    assert isinstance(optimizer, TikiTakaOptimizer)

    inputs = torch.tensor(
        [
            [0.8, -0.3],
            [-0.4, 0.9],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)
    initial_weights = [weight.state.clone() for weight in weights]

    for minibatch in range(2):
        optimizer.zero_grad()
        network.set_input(inputs, reset=True)
        free_minimizer.compute_equilibrium()
        cost_fn.set_target(labels)
        gradients = estimator.compute_gradient()

        assert len(gradients) == len(parameters)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        for parameter, gradient in zip(parameters, gradients):
            parameter.state.grad = gradient

        optimizer.step()
        for parameter in parameters:
            parameter.clamp_()

        if minibatch == 0:
            for weight, initial in zip(weights, initial_weights):
                torch.testing.assert_close(weight.state, initial)
            auxiliary_norm = sum(
                optimizer.auxiliary_state(weight.state).abs().sum()
                for weight in weights
            )
            assert auxiliary_norm > 0.0

    assert any(
        not torch.equal(weight.state, initial)
        for weight, initial in zip(weights, initial_weights)
    )
    for weight in weights:
        assert torch.isfinite(weight.state).all()
        torch.testing.assert_close(
            optimizer.auxiliary_state(weight.state),
            torch.zeros_like(weight.state),
        )
        state = optimizer.state[weight.state]
        assert state["step"] == 2
        assert state["transfer_count"] == 1
