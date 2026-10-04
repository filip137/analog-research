"""Optional historical gradient algorithms and trajectory diagnostics.

These utilities are outside the supported minibatch gradient interface. Their
historical numerical behavior is retained, including the recurrent algorithm's
requirement for a minimizer with a zero-argument ``step()`` (the current base
Minimizer requires a layer group). Moving them here does not validate those
research algorithms against the current solvers.
"""

from functools import singledispatch
from itertools import accumulate

import torch

from model.minimizer.minimizer import GradientDescentUpdater
from training.direct_readout import DirectReadoutGradient
from training.sgd import EquilibriumProp, Backprop, GradientEstimator


class RecurrentBackprop(GradientEstimator):
    """
    Class used to compute the parameter gradients of a cost function in an energy-based system via implicit differentiation (RecurrentBackprop).
    
    The algorithm implemented here is also known as `recurrent backpropagation', or the Almeida-Pineda algorithm.
    
    Attributes
    ----------
    _params (list of Parameters): the parameters whose gradients we want to estimate via implicit differentiation
    _layers (list of Layers): the Layers that minimize an energy function
    _cost_fn (CostFunction): the cost function to optimize
    energy_minimizer (EnergyMinimizer): the algorithm used to minimize the energy function

    Methods
    -------
    compute_gradient()
        Computes the parameter gradients via implicit differentiation
    detailed_gradients()
        Compute and return the sequence of time-dependent layer- and parameter- gradients via RecurrentBackprop
    """

    def __init__(self, params, layers, cost_fn, energy_minimizer):
        """Creates an instance of RecurrentBackprop

        Args:
            params (list of Parameters): the parameters whose gradients we want to estimate via equilibrium propagation
            layers (list of Layers): the Layers that minimize an energy function
            cost_fn (CostFunction): the cost function to optimize
            energy_minimizer (EnergyMinimizer): the algorithm used to minimize the energy function
        """

        self._params = params
        self._layers = layers
        self._cost_fn = cost_fn
        self._energy_minimizer = energy_minimizer

    def compute_gradient(self):
        """Computes the parameter gradients via implicit differentiation
        
        To compute the correct gradients, the layers must be in their 'free state' (equilibrium) when calling the method

        Returns:
            param_grads: list of Tensor of shape param_shape and type float32. The parameter gradients
        """

        # TODO: we use all the layers (including the input layer). Need to check if this is fine

        # TODO: this implementation is likely suboptimal in the case of e.g. the Readout cost function
        cost_grads = [self._cost_fn._grad(param, mean=True) for param in self._cost_fn.params()]  # compute the direct gradient of C, if C explicitly depends on parameters

        equilibrium_state = [layer.state for layer in self._layers]  # we store the state of the layers at equilibrium

        for layer in self._layers: layer.state.requires_grad = True
        for param in self._params: param.state.requires_grad = True

        # Compute the layer gradients at time t=0
        loss = self._cost_fn.eval().mean()
        layer_grads = torch.autograd.grad(loss, equilibrium_state, allow_unused=True, retain_graph=True)
        layer_grads = [torch.zeros_like(layer.state) if grad == None else grad for layer, grad in zip(self._layers, layer_grads)]
        param_grads = [torch.zeros_like(param.state) for param in self._params]

        # Compute the parameter gradients at time t>0
        self._energy_minimizer.step()
        next_state = [layer.state for layer in self._layers]

        for _ in range(self._energy_minimizer.num_iterations):
            # performs one step of Recurrent Backpropagation (the Almeida-Pineda algorithm) to compute the time-dependent layer- and parameter- gradients
            layer_grads_next = torch.autograd.grad(next_state, equilibrium_state, grad_outputs = layer_grads, retain_graph=True)
            param_grads_next = torch.autograd.grad(next_state, [param.state for param in self._params], grad_outputs = layer_grads, retain_graph=True)
            layer_grads = layer_grads_next
            param_grads = [param + param_next for param, param_next in zip(param_grads, param_grads_next)]
            
        for layer in self._layers:
            layer.state = layer.state.detach()
            layer.state.requires_grad = False
        for param in self._params: param.state.requires_grad = False

        return list(param_grads) + cost_grads

    def detailed_gradients(self, cumulative=True):
        """Compute and return the sequence of time-dependent layer- and parameter- gradients via RecurrentBackprop

        To compute the correct gradients, the layers must be at equilibrium when calling the method

        Args:
            cumulative (bool, optional): if True, computes the cumulative gradients ; if False, computes the gradients increases. Default: True.

        Returns:
            grads: dictionary of Tensor of shape variable_shape and type float32. The time-dependent gradients wrt the variables (layers and parameters)
        """

        equilibrium_state = [layer.state for layer in self._layers]  # we store the state of the layers at equilibrium

        grads = dict()  # dictionary of time-dependent layer- and parameter- gradients
        for layer in self._layers: grads[layer.name] = list()
        for param in self._params: grads[param.name] = list()

        for layer_state in equilibrium_state: layer_state.requires_grad = True
        for param in self._params: param.state.requires_grad = True

        # Compute the layer gradients at time t=0
        loss = self._cost_fn.eval().mean()
        layer_grads = torch.autograd.grad(loss, equilibrium_state, allow_unused=True, retain_graph=True)

        for layer, grad in zip(self._layers, layer_grads):
            if grad == None: grad = torch.zeros_like(layer.state)  # if the grad is None, then the corresponding parameter is not a part of the computational graph, therefore its gradient is zero
            grads[layer.name].append(grad)

        # Compute the layer- and parameter- gradients at time t>0
        self._energy_minimizer.step()
        next_state = [layer.state for layer in self._layers]

        for iteration in range(self._energy_minimizer.num_iterations):
            # performs one step of Recurrent Backpropagation (the Almeida-Pineda algorithm) to compute the time-dependent layer- and parameter- gradients
            layer_grads = [grads[layer.name][-1] for layer in self._layers]
            layer_grads_next = torch.autograd.grad(next_state, equilibrium_state, grad_outputs = layer_grads, retain_graph=True)
            param_grads_next = torch.autograd.grad(next_state, [param.state for param in self._params], grad_outputs = layer_grads, retain_graph=True)
            if iteration < self._energy_minimizer.num_iterations-1:
                for layer, grad in zip(self._layers, layer_grads_next): grads[layer.name].append(grad)
            for param, grad in zip(self._params, param_grads_next): grads[param.name].append(grad)

        # We reset the state of the layers as it was initially
        for layer, state in zip(self._layers, equilibrium_state):
            layer.state = state
            layer.state.requires_grad = False
        for param in self._params:
            param.state.requires_grad = False

        # We transform the values of the grads dictionary into Tensors
        if cumulative: grads = {name: list(accumulate(sequence)) for name, sequence in grads.items()}

        return grads

    def __str__(self):
        return 'Recurrent backpropagation'


class ContrastiveLearning(EquilibriumProp):
    """
    Class for estimating the parameter gradients of a cost function in an energy-based system via contrastive learning (CL)

    This class more generally implements the perturbation method of 'coupled learning', a modified version of contrastive learning and equilibrium propagation with weakly clamped outputs in the training phase.
    Since equilibrium propagation, contrastive learning and coupled learning are algorithmically similar, the ContrastiveLearning class inherits from the EquilibriumProp class and overrides the perturbation method, i.e. the difference between the two algorithms.

    This implementation of contrastive learning is generalized to any cost function (not just the MSE)
    """

    def __init__(self, params, layers, energy_fn, cost_fn, energy_minimizer, variant='positive', nudging=1.0):
        """Creates an instance of contrastive learning

        Args:
            params (list of Parameters): the parameters whose gradients we want to estimate via equilibrium propagation
            layers (list of Layers): the Layers that minimize the energy function
            energy_fn (Function): the energy function
            cost_fn (Function): the cost function
            energy_minimizer (Minimizer): the algorithm used to minimize the energy function
            variant (str, optional): either 'positive' (positively-perturbed CpL), 'negative' (negatively-perturbed CpL) or 'centered' (centered CpL). Default: 'positive'
            nudging (float, optional): the nudging value used to estimate the parameter gradients via EP. Default: 1.0
        """

        EquilibriumProp.__init__(self, params, layers, energy_fn, cost_fn, energy_minimizer, variant, nudging, use_alternative_formula=False)

        self._cost_fn = cost_fn

    def compute_gradient(self):
        """Calculates the direction of the weight updates of contrastive learning

        The weight updates depends on the coupled learning variant (positively-perturbed CpL, negatively-perturbed CpL, or centered CpL) and the nudging strength.
        
        Returns:
            param_grads: list of Tensor of shape param_shape and type float32. The directions of the weight updates
        """

        # TODO: this implementation is likely suboptimal in the case of e.g. the Readout cost function
        cost_grads = [self._cost_fn._grad(param, mean=True) for param in self._cost_fn.params()]  # compute the direct gradient of C, if C explicitly depends on parameters

        # First phase: compute the first equilibrium state of the layers
        layers_free = [layer.state for layer in self._layers]  # hack: we store the `free state' (i.e. the equilibrium state of the layers with nudging=0)
        output_gradients = self._cost_fn._grad(self._layers[-1], mean=False)  # FIXME
        self._layers[-1].state = self._layers[-1].state - self._first_nudging * output_gradients  # FIXME
        layers_first = self._energy_minimizer.compute_equilibrium()
        
        # Second phase: compute the second equilibrium state of the layers
        for layer, state in zip(self._layers, layers_free): layer.state = state  # hack: we start the second phase from the `free state' again
        output_gradients = self._cost_fn._grad(self._layers[-1], mean=False)  # FIXME
        self._layers[-1].state = self._layers[-1].state - self._second_nudging * output_gradients  # FIXME
        layers_second = self._energy_minimizer.compute_equilibrium()

        param_grads = self._standard_param_grads(layers_first, layers_second)

        return param_grads + cost_grads

    def __str__(self):
        return 'Contrastive learning -- mode={}, nudging={}'.format(self._variant, self._nudging)


@singledispatch
def detailed_gradients(estimator, cumulative=True):
    """Return historical trajectories for a supported research estimator."""
    raise TypeError(f"No detailed-gradient implementation for {type(estimator).__name__}.")


@detailed_gradients.register(EquilibriumProp)
def _ep_detailed_gradients(estimator, cumulative=True):
    """Compute and return the sequence of time-dependent layer- and parameter- EP gradients

    Calling this method leaves the state of the layers unchanged
    For the method to return the correct detailed gradients of EP, the layers must be at their 'free state' (equilibrium for nudging=0) when calling the method

    Args:
        cumulative (bool, optional): if True, computes the cumulative gradients ; if False, computes the gradients increases. Default: True.

    Returns:
        grads: dictionary of Tensor of shape variable_shape and type float32. The time-dependent gradients wrt the variables (layers and parameters)
    """

    layer_updaters = [
        GradientDescentUpdater(layer, estimator._augmented_fn)
        for layer in estimator._layers
    ]

    # First phase: compute the layers' activations along the first trajectory
    layers_free = [layer.state for layer in estimator._layers]  # we store the `free state' (i.e. the equilibrium state of the layers with nudging=0)
    if hasattr(estimator._augmented_fn, "prepare_nudging"):
        estimator._augmented_fn.prepare_nudging()
    estimator._augmented_fn.nudging = estimator._first_nudging
    trajectory_first = estimator._energy_minimizer.compute_trajectory()

    # Second phase: compute the layers' activations along the second trajectory
    for layer, state in zip(estimator._layers, layers_free): layer.state = state  # we start over from the `free state' again
    estimator._augmented_fn.nudging = estimator._second_nudging
    trajectory_second = estimator._energy_minimizer.compute_trajectory()

    # Compute the layer gradients
    trajectory_first = [dict(zip(trajectory_first, v)) for v in zip(*trajectory_first.values())]  # transform the dictionary of lists into a list of dictionaries
    trajectory_second = [dict(zip(trajectory_second, v)) for v in zip(*trajectory_second.values())]  # transform the dictionary of lists into a list of dictionaries
    layer_grads = [_ep_layer_grads(estimator, first, second, layer_updaters) for first, second in zip(trajectory_first[:-1], trajectory_second[:-1])]
    layer_grads = list(map(list, zip(*layer_grads)))  # transpose the list of lists: transform the time-wise layer-wise gradients into layer-wise time-wise gradients

    # Compute the parameter gradients with either the standard EquilibriumProp formula, or the alternative EquilibriumProp formula
    if estimator._use_alternative_formula:
        param_grads = [estimator._alternative_param_grads(layers_free, first, second) for first, second in zip(trajectory_first[1:], trajectory_second[1:])]
    else:
        param_grads = [estimator._standard_param_grads(first, second) for first, second in zip(trajectory_first[1:], trajectory_second[1:])]
    param_grads = list(map(list, zip(*param_grads)))  # transpose the list of lists: transform the time-wise parameter-wise gradients into parameter-wise time-wise gradients
    param_grads = [
        [grad * estimator._amplified_current_bias_gradient_scale(param) for grad in gradients]
        for param, gradients in zip(estimator._params, param_grads)
    ]

    # Store the layer-wise and parameter-wise time-wise gradients in a dictionary
    grads = dict()
    for layer, gradients in zip(estimator._layers, layer_grads): grads[layer.name] = gradients
    for param, gradients in zip(estimator._params, param_grads): grads[param.name] = gradients

    # Transform the time-wise gradients into time-wise increases if required
    if not cumulative:
        for layer in estimator._layers:
            layer_grads = grads[layer.name]
            grads[layer.name] = [layer_grads[0]] + [j-i for i, j in zip(layer_grads[:-1], layer_grads[1:])]
        for param in estimator._params:
            param_grads = grads[param.name]
            grads[param.name] = [param_grads[0]] + [j-i for i, j in zip(param_grads[:-1], param_grads[1:])]

    # Reset the layers to their `free state' values, where they were initially
    for layer, state in zip(estimator._layers, layers_free): layer.state = state

    return grads


def _ep_layer_grads(estimator, layers_first, layers_second, layer_updaters):
    """Compute the layer gradients given the activations of the first and second states

    Args:
        layers_first (dictionary of Tensors): the activations of the layers at the first state
        layers_second (dictionary of Tensors): the activations of the layers at the second state

    Returns:
        layer_grads: list of Tensors. The layer gradients
    """

    # FIXME: the gradients of the output layer are wrong because we need to set the correct 'output force' (nudging) at each time step.

    # Compute the energy gradients of the first state
    for layer in estimator._layers: layer.state = layers_first[layer.name]
    grads_first = [updater.grad() for updater in layer_updaters]

    # Compute the energy gradients of the second state
    for layer in estimator._layers: layer.state = layers_second[layer.name]
    grads_second = [updater.grad() for updater in layer_updaters]

    # Compute the layer gradients
    batch_size = estimator._layers[0].state.size(0)
    layer_grads = [(second - first) / ((estimator._second_nudging - estimator._first_nudging) * batch_size) for first, second in zip(grads_first, grads_second)]

    return layer_grads


@detailed_gradients.register(Backprop)
def _backprop_detailed_gradients(estimator, cumulative=True):
    """Compute and return the sequence of time-dependent layer- and parameter- gradients via Backprop.

    For the matching gradient property (MGP) to be satisfied, the layers must be at equilibrium when calling the method.

    Args:
        cumulative (bool, optional): if True, computes the cumulative gradients ; if False, computes the gradients increases. Default: True.

    Returns:
        grads: dictionary of Tensor of shape variable_shape and type float32. The time-dependent gradients wrt the variables (layers and parameters)
    """

    # We store the initial state of the layers and parameters, so we can reset them at the end of the method
    layers_state = [layer.state for layer in estimator._layers]
    params_state = [param.state for param in estimator._params]

    for param in estimator._params: param.state.requires_grad = True
    for layer in estimator._layers: layer.state.requires_grad = True

    # During the forward pass, we keep all the tensors of the computational graph in a 'trajectories' dictionary
    trajectories = estimator._energy_minimizer.compute_trajectory()

    # We perform the backward pass, and before that we ask all the variables (layers and parameters) to retain the partial deriavtive computed in the backward pass
    for trajectory in trajectories.values():
        for variable in trajectory:
            variable.retain_grad()
    loss = estimator._cost_fn.eval().mean()
    loss.backward()

    # After the backward pass, we read all the gradients in the .grad attributes of the variables (layers and parameters)
    # and we store these gradients in the 'grads' dictionary
    grads = dict()
    for layer in estimator._layers:
        layer_grads = [state.grad if state.grad is not None else torch.zeros_like(state) for state in trajectories[layer.name]]
        layer_grads = list(reversed(layer_grads))
        if cumulative: layer_grads = list(accumulate(layer_grads))
        grads[layer.name] = layer_grads[:-1]
    for param in estimator._params:
        param_grads = [state.grad if state.grad is not None else torch.zeros_like(state) for state in trajectories[param.name]]
        param_grads = list(reversed(param_grads))
        if not cumulative: param_grads = [param_grads[0]] + [j-i for i, j in zip(param_grads[:-1], param_grads[1:])]
        grads[param.name] = param_grads[:-1]

    # Finally, we reset the state of the layers and parameters where they were initially
    for layer, state in zip(estimator._layers, layers_state): layer.state = state
    for param, state in zip(estimator._params, params_state): param.state = state

    for layer in estimator._layers: layer.state.requires_grad = False
    for param in estimator._params: param.state.requires_grad = False

    return grads


@detailed_gradients.register(RecurrentBackprop)
def _recurrent_detailed_gradients(estimator, cumulative=True):
    return estimator.detailed_gradients(cumulative=cumulative)


@detailed_gradients.register(DirectReadoutGradient)
def _readout_detailed_gradients(estimator, cumulative=True):
    return {
        parameter.name: [gradient]
        for parameter, gradient in zip(
            estimator._cost_fn.params(), estimator.compute_gradient()
        )
    }
