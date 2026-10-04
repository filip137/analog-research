"""Core minibatch gradient estimators and EP nudging functions.

Optimizer updates and accumulation live outside this module. Optional historical
algorithms and trajectory inspection live in training.research_gradients.
"""

from abc import ABC, abstractmethod
import math
from numbers import Real
import torch

from model.function.interaction import Function, SumSeparableFunction
from model.minimizer.minimizer import ParamUpdater
from model.variable.layer import layer_index


def _amplified_layer_row_scale(energy_fn, layer):
    if getattr(energy_fn, "_differential_dense_edges", ()):
        scale_fn = getattr(energy_fn, "layer_energy_scale", None)
        if not callable(scale_fn):
            raise ValueError(
                "Expected a differential energy to expose a positive finite "
                f"layer metric. Provided value: {energy_fn!r}."
            )
        scale = float(scale_fn(layer))
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "Expected a differential energy to expose a positive finite "
                f"layer metric. Provided value: {scale!r}."
            )
        return scale
    voltage_amp = getattr(energy_fn, "_voltage_amp", getattr(energy_fn, "voltage_amp", None))
    current_amp = getattr(energy_fn, "_current_amp", getattr(energy_fn, "current_amp", None))
    if voltage_amp in (None, 0.0) or current_amp is None:
        return 1.0
    return float(voltage_amp / current_amp) ** max(layer_index(layer) - 1, 0)


def _resolve_current_scale(energy_fn, cost_fn, mode, current_scale):
    if mode != "current":
        return 1.0
    if current_scale is None:
        return 1.0
    if isinstance(current_scale, str):
        if current_scale == "auto":
            layers = cost_fn.layers()
            if not layers:
                return 1.0
            return _amplified_layer_row_scale(energy_fn, layers[-1])
        if current_scale in ("none", "legacy"):
            return 1.0
    return float(current_scale)


class Nudging(Function):
    """Class to apply either cost-based or current-based nudging.

    Attributes
    ----------
    _function (Function): the function to scale by a nudging factor
    nudging (float): the nudging value
    """

    # FIXME: how to deal with the case where the class is a QFunction or LFunction?

    def __init__(self, function, mode='cost', current_scale=1.0):
        """Initializes an instance of Nudging

        Args:
            function (Function): the function to scale by a nudging factor
        """
        self._function = function
        self._mode = mode
        self._nudging = 0.
        self._output_layer = function.layers()[-1] if function.layers() else None
        self._force = None
        self._current_scale = float(current_scale)

        if mode not in ('cost', 'current'):
            raise ValueError("expected nudging mode 'cost' or 'current', but got {}".format(mode))

        if mode == 'cost':
            layers = function.layers()
            params = function.params()
        else:
            layers = [] if self._output_layer is None else [self._output_layer]
            params = []

        Function.__init__(self, layers, params)

    @property
    def nudging(self):
        """Get and sets the nudging value"""
        return self._nudging

    @nudging.setter
    def nudging(self, nudging):
        self._nudging = nudging

    @property
    def mode(self):
        return self._mode

    def prepare(self):
        """Prepare the nudging term at the current free state.

        In ``cost`` mode this is a no-op because the nudging term is the full
        augmented cost ``beta * C``.
        In ``current`` mode we freeze the output force to the free-phase cost
        gradient so that only the linear coefficient ``b`` changes during the
        nudged phases.
        """

        if self._mode != 'current':
            return

        if self._output_layer is None:
            raise RuntimeError("current nudging requires a cost/output layer")

        # Force-based nudging uses the free-phase output gradient as a fixed
        # external current. The linear term is -<y, F>, so we store F = -dC/dy.
        self._force = - self._function._grad(self._output_layer, mean=False).detach().clone()

    def _force_state(self):
        if self._output_layer is None:
            raise RuntimeError("nudging has no managed output layer")
        if self._force is None or self._force.shape != self._output_layer.state.shape:
            return torch.zeros_like(self._output_layer.state)
        return self._force

    def eval(self):
        """Value of the nudging function. This is the function's value times the nudging value.

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the value of an example in the current mini-batch
        """
        if self._mode == 'cost':
            return self._nudging * self._function.eval()

        force = self._force_state()
        return - self._current_scale * self._nudging * self._output_layer.state.mul(force).flatten(start_dim=1).sum(dim=1)

    def grad_layer_fn(self, layer):
        if self._mode == 'cost':
            grad_layer_fn = self._function.grad_layer_fn(layer)
            return lambda: self._nudging * grad_layer_fn()

        dictionary = {self._output_layer: self._b_coef_output}
        return dictionary[layer]

    def grad_param_fn(self, param):
        if self._mode == 'cost':
            grad_param_fn = self._function.grad_param_fn(param)
            return lambda: self._nudging * grad_param_fn()

        return lambda: torch.zeros_like(param.state)

    def a_coef_fn(self, layer):
        """Returns the function that computes the coefficient a for a given layer"""
        if self._mode == 'cost':
            a_coef_fn = self._function.a_coef_fn(layer)
            return lambda: self._nudging * a_coef_fn()
        return lambda: 0.

    def b_coef_fn(self, layer):
        """Returns the function that computes the coefficient b for a given layer"""
        if self._mode == 'cost':
            b_coef_fn = self._function.b_coef_fn(layer)
            return lambda: self._nudging * b_coef_fn()
        dictionary = {self._output_layer: self._b_coef_output}
        return dictionary[layer]

    def _b_coef_output(self):
        return - self._current_scale * self._nudging * self._force_state()


class AugmentedFunction(SumSeparableFunction):
    """
    Class to augment an 'energy' function by a 'cost' function.

    Attributes
    ----------
    nudging (float): nudging value

    Methods
    -------
    eval()
        Returns the value of the augmented function (for the current configuration)
    """

    def __init__(self, energy_fn, cost_fn, nudging_mode='cost', current_scale='auto'):
        """Creates an instance of AugmentedFunction"""

        layers = energy_fn.layers()
        params = energy_fn.params()

        resolved_current_scale = _resolve_current_scale(
            energy_fn,
            cost_fn,
            nudging_mode,
            current_scale,
        )
        nudging = Nudging(cost_fn, mode=nudging_mode, current_scale=resolved_current_scale)
        interactions = [energy_fn, nudging]

        SumSeparableFunction.__init__(self, layers, params, interactions)
        self._nudging = nudging
        self._energy_fn = energy_fn
        self._nudging_mode = nudging_mode
        self._current_scale = resolved_current_scale
        self._amplified_current_correction_enabled = (
            nudging_mode == 'current' and resolved_current_scale != 1.0
        )

        # FIXME: what if the cost function does not have the same layers and/or params as the energy function?


    @property
    def nudging(self):
        """Get and sets the nudging value"""
        return self._nudging.nudging

    @nudging.setter
    def nudging(self, nudging):
        self._nudging.nudging = nudging

    @property
    def nudging_mode(self):
        return self._nudging_mode

    @property
    def amplified_current_correction_enabled(self):
        return self._amplified_current_correction_enabled

    def prepare_nudging(self):
        self._nudging.prepare()

    def eval(self):
        """Returns the value of the augmented function for the current configuration.

        Returns:
            Tensor of shape (batch_size,) and type float32. Vector of values for each of the examples in the current mini-batch
        """

        return self._energy_fn.eval() + self._nudging.eval()


def _validated_nudging(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"Expected finite positive nudging, got {value!r}.")
    return value


def _validated_variant(value):
    if value not in ('positive', 'negative', 'centered'):
        raise ValueError(f"Expected EP variant positive, negative, or centered, got {value!r}.")
    return value


def _validated_alternative_formula(value):
    if not isinstance(value, bool):
        raise ValueError(f"Expected a boolean use_alternative_formula, got {value!r}.")
    return value


class GradientEstimator(ABC):
    """
    Abstract class for computing or estimating the parameter gradients in a bilevel optimization problem

    A bilevel optimization problem is a problem of the form:

    minimize C(s(theta))
    subject to s(theta) = argmin_s E(theta,s)

    where theta and s are variables, and C(s) and E(theta,s) are scalar functions. Specifally:
    - theta is a set of 'parameter variables'
    - s is a set of 'layer variables'
    - C(s) is a 'cost function' 
    - E(theta,s) is an 'energy function'

    This abstract class allows to compute (or estimate) the gradient of C(s(theta)) wrt theta. It is used in particular:
    - to perform SGD (stochastic gradient descent) in an energy-based system,
    - to check the matching gradients property (MGP) in an energy-based system.

    Methods
    -------
    compute_gradient()
        Compute and return the parameter gradients
    """

    @abstractmethod
    def compute_gradient(self):
        """Compute the parameter gradients"""
        pass


class EquilibriumProp(GradientEstimator):
    """
    Class for estimating the parameter gradients of a cost function in an energy-based system via equilibrium propagation (EP)

    Equilibrium propagation (EP) estimates the gradient (wrt to theta) of C(s(theta)) under the constraint that s(theta) = argmin_s E(theta,s)

    The EP gradient estimator depends on two scalars: nudging_1 and nudging_2. It is given by the formula
    estimator(nudging_1, nudging_2) := [ dE(theta,s(nudging_2))/dtheta - dE(theta,s(nudging_1))/dtheta ] / [ nudging_2 - nudging_1 ]
    where s(nudging) = argmin_s [ E(theta,s) + nudging * C(s) ]

    The scalars nudging_1 and nudging_2 are determined by the attributes 'variant' and 'nudging' of the EquilibriumProp class, as follows:
    * if the variant is 'positive', then nudging_1 = 0 and nudging_2 = + nudging
    * if the variant is 'negative', then nudging_1 = - nudging and nudging_2 = 0
    * if the variant is 'centered', then nudging_1 = - nudging and nudging_2 = + nudging

    Attributes
    ----------
    _params (list of Parameters): the parameters whose gradients we want to estimate via equilibrium propagation
    _layers (list of Layers): the layers that minimize the augmented energy function E(theta,s) + nudging * C(s)
    _energy_minimizer (Minimizer): the algorithm used to minimize the augmented energy function (in the perturbed phase of EP)
    variant (str): either 'positive' (positively-perturbed EP), 'negative' (negatively-perturbed EP) or 'centered' (centered EP)
    nudging (float): the nudging value used to estimate the parameter gradients via EP
    use_alternative_formula (bool): which equilibrium propagation formula is used to estimating the parameter gradients. Either the 'standard' formula (False) or the 'alternative' formula (True)

    Methods
    -------
    compute_gradient()
        Compute and return the parameter gradients via EP
    """

    def __init__(self, params, layers, energy_fn, cost_fn, energy_minimizer, variant='centered', nudging=0.25, use_alternative_formula=False):
        """Creates an instance of equilibrium propagation

        Args:
            params (list of Parameters): the parameters whose gradients we want to estimate via equilibrium propagation
            layers (list of Layers): the Layers that minimize the augmented energy function
            energy_fn (Function): the energy function
            cost_fn (Function): the cost function
            energy_minimizer (Minimizer): the algorithm used to minimize the augmented energy function
            variant (str, optional): either 'positive' (positively-perturbed EP), 'negative' (negatively-perturbed EP) or 'centered' (centered EP). Default: 'centered'
            nudging (float, optional): the nudging value used to estimate the parameter gradients via EP. Default: 0.25
            use_alternative_formula (bool, optional): which equilibrium propagation formula is used to estimating the parameter gradients. Either the 'standard' formula (False) or the 'alternative' formula (True). Default: False
        """

        self._params = params
        self._layers = layers

        self._param_updaters = [ParamUpdater(param, energy_fn) for param in params]

        self._augmented_fn = energy_fn
        self._energy_minimizer = energy_minimizer
        self._cost_fn = cost_fn

        self._nudging = _validated_nudging(nudging)
        self._variant = _validated_variant(variant)
        self._set_nudgings()

        self._use_alternative_formula = _validated_alternative_formula(use_alternative_formula)

    @property
    def nudging(self):
        """Get and sets the nudging value used for estimating the parameter gradients via equilibrium propagation"""
        return self._nudging

    @nudging.setter
    def nudging(self, nudging):
        self._nudging = _validated_nudging(nudging)
        self._set_nudgings()

    @property
    def variant(self):
        """Get and sets the training mode ('positive', 'negative' or 'centered')"""
        return self._variant

    @variant.setter
    def variant(self, variant):
        self._variant = _validated_variant(variant)
        self._set_nudgings()

    @property
    def use_alternative_formula(self):
        """Get and sets the use_alternative_formula attribute"""
        return self._use_alternative_formula

    @use_alternative_formula.setter
    def use_alternative_formula(self, use_alternative_formula):
        self._use_alternative_formula = _validated_alternative_formula(use_alternative_formula)

    def compute_gradient(self):
        """Estimates the parameter gradient via equilibrium propagation

        The gradient depends on the EP variant (positively-perturbed EP, negatively-perturbed EP, or centered EP) and the nudging strength.
        To get a better gradient estimate, this method must be called when the layers are at their 'free state' (equilibrium for nudging=0).
        
        Returns:
            param_grads: list of Tensor of shape param_shape and type float32. The parameter gradients
        """

        # TODO: this implementation is likely suboptimal in the case of e.g. the Readout cost function
        cost_grads = [self._cost_fn._grad(param, mean=True) for param in self._cost_fn.params()]  # compute the direct gradient of C, if C explicitly depends on parameters
        
        # First phase: compute the first equilibrium state of the layers
        layers_free = [layer.state for layer in self._layers]  # hack: we store the `free state' (i.e. the equilibrium state of the layers with nudging=0)
        if hasattr(self._augmented_fn, "prepare_nudging"):
            self._augmented_fn.prepare_nudging()
        self._augmented_fn.nudging = self._first_nudging
        layers_first = self._energy_minimizer.compute_equilibrium()
        
        # Second phase: compute the second equilibrium state of the layers
        for layer, state in zip(self._layers, layers_free): layer.state = state  # hack: we start the second phase from the `free state' again
        self._augmented_fn.nudging = self._second_nudging
        layers_second = self._energy_minimizer.compute_equilibrium()

        # Compute the parameter gradients with either the standard EquilibriumProp formula, or the alternative EquilibriumProp formula
        if self._use_alternative_formula:
            param_grads = self._alternative_param_grads(layers_free, layers_first, layers_second)
        else:
            param_grads = self._standard_param_grads(layers_first, layers_second)
        param_grads = self._apply_amplified_current_bias_gradient_scale(param_grads)

        return param_grads + cost_grads


    def _standard_param_grads(self, layers_first, layers_second):
        """Compute the parameter gradients using the standard EquilibriumProp formula

        Args:
            layers_first (dictionary of Tensors): the activations of the layers at the first state
            layers_second (dictionary of Tensors): the activations of the layers at the second state

        Returns:
            param_grads: list of Tensors. The parameter gradients
        """

        # Compute the energy gradients of the first state
        for layer in self._layers: layer.state = layers_first[layer.name]
        grads_first = [updater.grad() for updater in self._param_updaters]

        # Compute the energy gradients of the second state
        for layer in self._layers: layer.state = layers_second[layer.name]
        grads_second = [updater.grad() for updater in self._param_updaters]

        # Compute the parameter gradients
        param_grads = [(second - first) / (self._second_nudging - self._first_nudging) for first, second in zip(grads_first, grads_second)]

        return param_grads

    def _amplified_current_bias_gradient_scale(self, param):
        if not getattr(self._augmented_fn, "amplified_current_correction_enabled", False):
            return 1.0
        energy_fn = getattr(self._augmented_fn, "_energy_fn", self._augmented_fn)
        voltage_amp = getattr(energy_fn, "_voltage_amp", getattr(energy_fn, "voltage_amp", None))
        current_amp = getattr(energy_fn, "_current_amp", getattr(energy_fn, "current_amp", None))
        if voltage_amp in (None, 0.0) or current_amp is None:
            return 1.0

        for interaction in getattr(energy_fn, "_interactions", []):
            if getattr(interaction, "_bias", None) is param:
                layer = getattr(interaction, "_layer", None)
                if layer is None:
                    return 1.0
                return float(current_amp / voltage_amp) ** max(layer_index(layer) - 1, 0)
        return 1.0

    def _apply_amplified_current_bias_gradient_scale(self, param_grads):
        return [
            grad * self._amplified_current_bias_gradient_scale(param)
            for param, grad in zip(self._params, param_grads)
        ]

    def _alternative_param_grads(self, layers_free, layers_first, layers_second):
        """Compute the parameter gradients using the alternative EquilibriumProp formula

        Args:
            layers_free (list of Tensors): the activations of the layers at the free state
            layers_first (dictionary of Tensors): the activations of the layers at the first state
            layers_second (dictionary of Tensors): the activations of the layers at the second state

        Returns:
            param_grads: list of Tensors. The parameter gradients
        """

        direction = {layer.name: (layers_second[layer.name] - layers_first[layer.name]) / (self._second_nudging - self._first_nudging) for layer in self._layers}
        for layer, state in zip(self._layers, layers_free): layer.state = state  # hack: we need to reset the state of the layers at the `free state'
        param_grads = [updater.second_fn(direction) for updater in self._param_updaters]

        return param_grads


    def _set_nudgings(self):
        """Sets the nudging values of the first and second states, depending on the attributes variant and nudging

        first_nudging: nudging value used to compute the first state of equilibrium propagation
        second_nudging: nudging value used to compute the second state of equilibrium propagation
        """

        if self._variant == "positive":
            self._first_nudging = 0.
            self._second_nudging = self._nudging
        elif self._variant == "negative":
            self._first_nudging = -self._nudging
            self._second_nudging = 0.
        elif self._variant == "centered":
            self._first_nudging = -self._nudging
            self._second_nudging = self._nudging
        else:
            raise ValueError("expected 'positive', 'negative' or 'centered' but got {}".format(self._variant))

    def __str__(self):
        formula = 'alternative' if self._use_alternative_formula else 'standard'
        return 'Equilibrium propagation -- mode={}, nudging={}, formula={}'.format(self._variant, self._nudging, formula)


class Backprop(GradientEstimator):
    """
    Class used to compute the parameter gradients of a cost function in an energy-based system via backpropagation (automatic differentiation)

    Attributes
    ----------
    _params (list of Parameters): the parameters whose gradients we want to estimate via backpropagation
    _layers (list of Layers): the layers that minimize the augmented energy function
    cost_fn (CostFunction): the cost function to optimize
    energy_minimizer (Minimizer): the algorithm used to minimize the augmented energy function

    Methods
    -------
    compute_gradient()
        Computes the parameter gradients via backpropagation
    """

    def __init__(self, params, layers, cost_fn, energy_minimizer):
        """Creates an instance of backpropagation

        Args:
            params (list of Parameters): the parameters whose gradients we want to estimate via backpropagation
            layers (list of Layers): the Layers that minimize the augmented energy function
            cost_fn (CostFunction): the cost function whose parameter gradients we want to compute
            energy_minimizer (Minimizer): the algorithm used to minimize the augmented energy function
        """

        self._params = params
        self._layers = layers
        self._cost_fn = cost_fn
        self._energy_minimizer = energy_minimizer

    def compute_gradient(self):
        """Computes the parameter gradients via backpropagation
        
        Returns:
            param_grads: list of Tensor of shape param_shape and type float32. The parameter gradients
        """

        # TODO: this implementation is likely suboptimal in the case of e.g. the Readout cost function
        cost_grads = [self._cost_fn._grad(param, mean=True) for param in self._cost_fn.params()]  # compute the direct gradient of C, if C explicitly depends on parameters

        states_and_flags = [(param.state, param.state.requires_grad) for param in self._params]
        try:
            for state, _ in states_and_flags:
                state.requires_grad_(True)
            self._energy_minimizer.compute_equilibrium()
            cost_mean = self._cost_fn.eval().mean()
            param_grads = torch.autograd.grad(cost_mean, [param.state for param in self._params])
            return list(param_grads) + cost_grads
        finally:
            for state, requires_grad in states_and_flags:
                state.requires_grad_(requires_grad)
            for layer in self._layers:
                layer.state = layer.state.detach()


    def __str__(self):
        return 'Backpropagation'
