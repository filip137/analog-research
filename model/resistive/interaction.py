from abc import ABC, abstractmethod
import math
from numbers import Integral, Real
import torch
import torch.nn.functional as F

from model.function.interaction import QFunction
from model.variable.layer import LinearLayer, layer_index


class DenseResistive(QFunction):
    """Dense edge with energy 0.5 * D * g * (alpha * v_pre - v_post)^2.

    D = (current_amp / voltage_amp)^pre_index, and alpha is one at the
    clamped input and voltage_amp elsewhere. All coefficients and parameter
    gradients use this same energy convention.

    Attributes
    ----------
    _layer_pre (Layer): pre-synaptic layer.
    _layer_post (Layer): post-synaptic layer.
    _weight (DenseWeight): weight tensor between layer_pre and layer_post. Tensor of shape (layer_pre_shape, layer_post_shape). Type is float32.
    """

    def __init__(
        self,
        layer_pre,
        layer_post,
        dense_weight,
        voltage_amp,
        current_amp,
        *,
        logical_pre_index=None,
        logical_post_index=None,
    ):
        """Initializes an instance of DenseResistive

        Args:
            layer_pre (Layer): pre-synaptic layer
            layer_post (Layer): post-synaptic layer
            dense_weight (DenseWeight): weight tensor between layer_pre and layer_post. Tensor of shape (layer_pre_shape, layer_post_shape). Type is float32.

        """

        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight = dense_weight
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp
        self._logical_pre_index = (
            layer_index(self._layer_pre)
            if logical_pre_index is None
            else int(logical_pre_index)
        )
        self._logical_post_index = (
            layer_index(self._layer_post)
            if logical_post_index is None
            else int(logical_post_index)
        )

        QFunction.__init__(self, [layer_pre, layer_post], [dense_weight])

    def _amp_factor(self):
        return (self._current_amp / self._voltage_amp) ** self._logical_pre_index

    def _pre_scale(self):
        return 1.0 if self._logical_pre_index == 0 else self._voltage_amp

    def eval(self):
        """Computes the energy term corresponding to this weight tensor.

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy term of an example in the current mini-batch
        """

        layer_pre = self._layer_pre.state.clone()
        if self._logical_pre_index != 0:
            layer_pre = layer_pre * self._voltage_amp
        layer_post = self._layer_post.state  # / self._layer_post.gain
        layer_post = layer_post
        dims_pre = len(self._layer_pre.shape)
        dims_post = len(self._layer_post.shape)
        for _ in range(dims_post): layer_pre = layer_pre.unsqueeze(-1)  # broadcast layer_pre to (batch_size, shape_pre, shape_post)
        for _ in range(dims_pre): layer_post = layer_post.unsqueeze(1)  # broadcast layer_post to (batch_size, shape_pre, shape_post)
        weight = self._weight.get().unsqueeze(0)  # broadcast weight to (batch_size, shape_pre, shape_post)
        return 0.5 * ((layer_pre - layer_post)**2).mul(weight).flatten(start_dim=1).sum(dim=1) * self._amp_factor()

    def a_coef_fn(self, layer):
        """Overrides the default implementation of QFunction"""
        dictionary = {
            self._layer_pre: self._a_coef_layer_pre,
            self._layer_post: self._a_coef_layer_post,
            }
        return dictionary[layer]

    def b_coef_fn(self, layer):
        """Overrides the default implementation of QFunction"""
        dictionary = {
            self._layer_pre: self._b_coef_layer_pre,
            self._layer_post: self._b_coef_layer_post,
            }
        return dictionary[layer]

    def grad_param_fn(self, param):
        """Overrides the default implementation of Function"""
        dictionary = {self._weight: self._grad_weight}
        return dictionary[param]

    def _b_coef_layer_pre(self):
        """Returns the interaction's linear influence on the pre-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float32: the linear contribution on layer_pre
        """

        layer_post = self._layer_post.state
        dims_pre = len(self._layer_pre.shape)
        dims_post = len(self._layer_post.shape)  # number of dimensions involved in the tensor product
        weight = self._weight.get()
        dim_weight = len(weight.shape)
        permutation = tuple(range(dims_pre, dim_weight)) + tuple(range(dims_pre))
        b_coef = - torch.tensordot(layer_post, weight.permute(permutation), dims=dims_post)
        b_coef = b_coef * self._amp_factor() * self._pre_scale()
        return b_coef

    def _a_coef_layer_pre(self):
        """Returns the interaction's linear influence on the pre-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float32: the linear contribution on layer_pre
        """

        dims_pre = len(self._layer_pre.shape)
        a_coef = 0.5 * self._weight.get().flatten(start_dim=dims_pre).sum(dim=-1).unsqueeze(0)

        a_coef = a_coef * self._amp_factor() * self._pre_scale() ** 2

        return a_coef

    def _b_coef_layer_post(self):
        """Returns the interaction's linear influence on the post-synaptic layer.
            this acccounts for the previous layer being amplified -- the inputs are not amplified
        Returns:
            Tensor of shape (batch_size, layer_post_shape) and type float32: the linear contribution on layer_post
        """

        layer_pre = self._layer_pre.state
        dims_pre = len(self._layer_pre.shape)  # number of dimensions involved in the tensor product
        b_coef = - torch.tensordot(layer_pre, self._weight.get(), dims=dims_pre)
        return b_coef * self._amp_factor() * self._pre_scale()

    def _a_coef_layer_post(self):
        """Returns the interaction's quadratic influence on the post-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_post_shape) and type float32: the quadratic contribution on layer_post
        """

        dims = len(self._layer_pre.shape) - 1
        a_coef = 0.5 * self._weight.get().flatten(end_dim=dims).sum(dim=0).unsqueeze(0)

        return a_coef * self._amp_factor()

    def _grad_weight(self):
        """Returns the interaction's gradient wrt the weight

        Returns:
            Tensor of shape weight_shape and type float32: the gradient wrt the weights
        """


        layer_pre = self._layer_pre.state.clone()
        if self._logical_pre_index != 0:
            layer_pre *= self._voltage_amp
        layer_post = self._layer_post.state
        dims_pre = len(self._layer_pre.shape)
        dims_post = len(self._layer_post.shape)
        for _ in range(dims_post): layer_pre = layer_pre.unsqueeze(-1)
        for _ in range(dims_pre): layer_post = layer_post.unsqueeze(1)
        grad_weight = 0.5 * ((layer_pre - layer_post)**2).mean(dim=0) * self._amp_factor()
        return grad_weight


class SignedDenseResistive(QFunction):
    """Dense resistive edge implemented by a differential device pair.

    The input edge has unit forward gain, while every interior edge has
    forward gain ``voltage_amp``.  Multiplying the two branch energies by the
    post-layer metric ``(current_amp / voltage_amp) ** logical_pre_index``
    produces a scalar energy whose layer gradients are the symmetrized port
    currents.  Consequently, signal transfer is controlled by ``G+ - G-``
    and loading/curvature by ``G+ + G-``.  Both tensors remain non-negative
    physical conductances.
    """

    def __init__(
        self,
        layer_pre,
        layer_post,
        conductance_plus,
        conductance_minus,
        voltage_amp,
        current_amp,
        *,
        logical_pre_index=None,
        logical_post_index=None,
    ):
        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._conductance_plus = conductance_plus
        self._conductance_minus = conductance_minus
        for name, conductance in (
            ("conductance_plus", conductance_plus),
            ("conductance_minus", conductance_minus),
        ):
            state = getattr(conductance, "state", None)
            clamp_min = getattr(conductance, "min_cond", None)
            clamp_max = getattr(conductance, "max_cond", None)
            effective_clamp_min = 0.0 if clamp_min is None else clamp_min
            if (
                not isinstance(state, torch.Tensor)
                or getattr(conductance, "_non_negative", False) is not True
                or (
                    clamp_min is not None
                    and (
                        isinstance(clamp_min, bool)
                        or not isinstance(clamp_min, Real)
                        or not math.isfinite(float(clamp_min))
                        or float(clamp_min) < 0.0
                    )
                )
                or (
                    clamp_max is not None
                    and (
                        isinstance(clamp_max, bool)
                        or not isinstance(clamp_max, Real)
                        or not math.isfinite(float(clamp_max))
                        or float(clamp_max) < float(effective_clamp_min)
                    )
                )
                or not torch.isfinite(state).all().item()
                or (state < 0.0).any().item()
            ):
                raise ValueError(
                    f"Expected {name} to be a finite, non-negative-clamped "
                    "physical conductance tensor. Provided value: "
                    f"parameter={conductance!r}, clamp_min={clamp_min!r}, "
                    f"clamp_max={clamp_max!r}."
                )
        if (
            isinstance(voltage_amp, bool)
            or not isinstance(voltage_amp, Real)
            or isinstance(current_amp, bool)
            or not isinstance(current_amp, Real)
        ):
            raise ValueError(
                "Expected voltage_amp and current_amp to be finite positive "
                "numbers. Provided value: "
                f"voltage_amp={voltage_amp!r}, current_amp={current_amp!r}."
            )
        resolved_voltage_amp = float(voltage_amp)
        resolved_current_amp = float(current_amp)
        if (
            not math.isfinite(resolved_voltage_amp)
            or resolved_voltage_amp <= 0.0
            or not math.isfinite(resolved_current_amp)
            or resolved_current_amp <= 0.0
        ):
            raise ValueError(
                "Expected voltage_amp and current_amp to be finite positive "
                "numbers. Provided value: "
                f"voltage_amp={voltage_amp!r}, current_amp={current_amp!r}."
            )
        self._voltage_amp = resolved_voltage_amp
        self._current_amp = resolved_current_amp
        try:
            if logical_pre_index is None or logical_post_index is None:
                raise TypeError
            if (
                isinstance(logical_pre_index, bool)
                or not isinstance(logical_pre_index, Integral)
            ):
                raise TypeError
            if (
                isinstance(logical_post_index, bool)
                or not isinstance(logical_post_index, Integral)
            ):
                raise TypeError
            self._logical_pre_index = int(logical_pre_index)
            self._logical_post_index = int(logical_post_index)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Expected explicit logical layer indices to be adjacent "
                "non-negative integers with logical_post_index = "
                "logical_pre_index + 1. "
                "Provided value: "
                f"logical_pre_index={logical_pre_index!r}, "
                f"logical_post_index={logical_post_index!r}."
            ) from error
        if (
            self._logical_pre_index < 0
            or self._logical_post_index != self._logical_pre_index + 1
        ):
            raise ValueError(
                "Expected explicit logical layer indices to be adjacent "
                "non-negative integers with logical_post_index = "
                "logical_pre_index + 1. "
                "Provided value: "
                f"logical_pre_index={logical_pre_index!r}, "
                f"logical_post_index={logical_post_index!r}, "
                f"resolved=({self._logical_pre_index!r}, "
                f"{self._logical_post_index!r})."
            )
        try:
            post_metric = (
                1.0
                if self._logical_pre_index == 0
                else math.pow(
                    self._current_amp / self._voltage_amp,
                    self._logical_pre_index,
                )
            )
        except (OverflowError, ZeroDivisionError, ValueError) as error:
            raise ValueError(
                "Expected amplifier magnitudes and logical depth to yield a "
                "finite positive post-layer energy metric. Provided value: "
                f"voltage_amp={voltage_amp!r}, current_amp={current_amp!r}, "
                f"logical_pre_index={self._logical_pre_index!r}."
            ) from error
        represented_metric = torch.as_tensor(
            post_metric,
            dtype=self._layer_post.state.dtype,
            device=self._layer_post.state.device,
        )
        if (
            not math.isfinite(post_metric)
            or post_metric <= 0.0
            or not torch.isfinite(represented_metric).item()
            or represented_metric.item() <= 0.0
        ):
            raise ValueError(
                "Expected amplifier magnitudes and logical depth to yield a "
                "finite positive post-layer energy metric. Provided value: "
                f"voltage_amp={voltage_amp!r}, current_amp={current_amp!r}, "
                f"logical_pre_index={self._logical_pre_index!r}, "
                f"dtype={self._layer_post.state.dtype!s}."
            )
        self._post_metric_value = post_metric
        forward_gain = (
            1.0
            if self._logical_pre_index == 0
            else self._voltage_amp
        )
        try:
            coefficient_scales = (
                forward_gain,
                post_metric * forward_gain,
                post_metric * forward_gain * forward_gain,
            )
        except OverflowError as error:
            raise ValueError(
                "Expected amplifier magnitudes and logical depth to yield "
                "finite positive energy coefficient scales. Provided value: "
                f"voltage_amp={voltage_amp!r}, current_amp={current_amp!r}, "
                f"logical_pre_index={self._logical_pre_index!r}."
            ) from error
        represented_scales = torch.as_tensor(
            coefficient_scales,
            dtype=self._layer_post.state.dtype,
            device=self._layer_post.state.device,
        )
        if (
            not all(
                math.isfinite(scale) and scale > 0.0
                for scale in coefficient_scales
            )
            or not torch.isfinite(represented_scales).all().item()
            or not (represented_scales > 0.0).all().item()
        ):
            raise ValueError(
                "Expected amplifier magnitudes and logical depth to yield "
                "finite positive energy coefficient scales. Provided value: "
                f"voltage_amp={voltage_amp!r}, current_amp={current_amp!r}, "
                f"logical_pre_index={self._logical_pre_index!r}, "
                f"dtype={self._layer_post.state.dtype!s}."
            )
        QFunction.__init__(
            self,
            [layer_pre, layer_post],
            [conductance_plus, conductance_minus],
        )

    def _sum(self):
        return self._conductance_plus.get() + self._conductance_minus.get()

    def _difference(self):
        return self._conductance_plus.get() - self._conductance_minus.get()

    def _forward_gain(self):
        return 1.0 if self._logical_pre_index == 0 else self._voltage_amp

    def _post_metric(self):
        return self._post_metric_value

    def _broadcast_states(self):
        layer_pre = self._layer_pre.state
        layer_post = self._layer_post.state
        dims_pre = len(self._layer_pre.shape)
        dims_post = len(self._layer_post.shape)
        for _ in range(dims_post):
            layer_pre = layer_pre.unsqueeze(-1)
        for _ in range(dims_pre):
            layer_post = layer_post.unsqueeze(1)
        return layer_pre, layer_post

    def eval(self):
        layer_pre, layer_post = self._broadcast_states()
        plus = self._conductance_plus.get().unsqueeze(0)
        minus = self._conductance_minus.get().unsqueeze(0)
        pre = self._forward_gain() * layer_pre
        energy = (pre - layer_post).square().mul(plus)
        energy = energy + (pre + layer_post).square().mul(minus)
        return (
            0.5
            * self._post_metric()
            * energy.flatten(start_dim=1).sum(dim=1)
        )

    def grad_layer_fn(self, layer):
        a_coef = self.a_coef_fn(layer)
        b_coef = self.b_coef_fn(layer)
        return lambda: 2.0 * a_coef() * layer.state + b_coef()

    def a_coef_fn(self, layer):
        return {
            self._layer_pre: self._a_coef_layer_pre,
            self._layer_post: self._a_coef_layer_post,
        }[layer]

    def b_coef_fn(self, layer):
        return {
            self._layer_pre: self._b_coef_layer_pre,
            self._layer_post: self._b_coef_layer_post,
        }[layer]

    def grad_param_fn(self, param):
        return {
            self._conductance_plus: self._grad_plus,
            self._conductance_minus: self._grad_minus,
        }[param]

    def _a_coef_layer_pre(self):
        dims_pre = len(self._layer_pre.shape)
        result = 0.5 * self._sum().flatten(
            start_dim=dims_pre
        ).sum(dim=-1).unsqueeze(0)
        return (
            result
            * self._post_metric()
            * self._forward_gain() ** 2
        )

    def _b_coef_layer_pre(self):
        layer_post = self._layer_post.state
        dims_pre = len(self._layer_pre.shape)
        dims_post = len(self._layer_post.shape)
        difference = self._difference()
        dim_weight = len(difference.shape)
        permutation = tuple(range(dims_pre, dim_weight)) + tuple(
            range(dims_pre)
        )
        result = -torch.tensordot(
            layer_post,
            difference.permute(permutation),
            dims=dims_post,
        )
        return result * self._post_metric() * self._forward_gain()

    def _a_coef_layer_post(self):
        dims = len(self._layer_pre.shape) - 1
        result = 0.5 * self._sum().flatten(
            end_dim=dims
        ).sum(dim=0).unsqueeze(0)
        return result * self._post_metric()

    def _b_coef_layer_post(self):
        layer_pre = self._layer_pre.state
        dims_pre = len(self._layer_pre.shape)
        result = -torch.tensordot(
            layer_pre,
            self._difference(),
            dims=dims_pre,
        )
        return result * self._post_metric() * self._forward_gain()

    def _grad_conductance(self, sign):
        layer_pre, layer_post = self._broadcast_states()
        return 0.5 * (
            self._forward_gain() * layer_pre + sign * layer_post
        ).square().mean(dim=0) * self._post_metric()

    def _grad_plus(self):
        return self._grad_conductance(-1.0)

    def _grad_minus(self):
        return self._grad_conductance(1.0)


class ConvResistive(QFunction):
    """Convolutional resistive interaction mirroring DenseResistive logic in conv form."""

    def __init__(self, layer_pre, layer_post, conv_weight, padding, stride, dilation, voltage_amp, current_amp):
        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight = conv_weight
        QFunction.__init__(self, [layer_pre, layer_post], [conv_weight])
        self._P = padding
        self._S = stride
        self._D = dilation
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp

    def _amp_factor(self):
        return (self._current_amp / self._voltage_amp) ** layer_index(self._layer_pre)

    def _pre_scale(self):
        return 1.0 if layer_index(self._layer_pre) == 0 else self._voltage_amp

    def _conv_geometry(self):
        weight = self._weight.get()
        C_out, C_in, Kh, Kw = weight.shape
        x = self._layer_pre.state
        _, _, H_in, W_in = x.shape

        H_out = (H_in + 2 * self._P - self._D * (Kh - 1) - 1) // self._S + 1
        W_out = (W_in + 2 * self._P - self._D * (Kw - 1) - 1) // self._S + 1

        return weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out

    def eval(self, per_sample=True):
        weight, C_out, C_in, Kh, Kw, *_ = self._conv_geometry()

        layer_pre = self._layer_pre.state * self._pre_scale()
        layer_post_scaled = self._layer_post.state


        cols = F.unfold(layer_pre, (Kh, Kw), padding=self._P, stride=self._S, dilation=self._D)
        N, C_out, H_out, W_out = layer_post_scaled.shape
        K = C_in * Kh * Kw
        L = H_out * W_out

        patches = cols.transpose(1, 2).unsqueeze(2)
        kernels = weight.view(1, 1, C_out, K)
        targets = layer_post_scaled.view(N, C_out, L).transpose(1, 2).unsqueeze(-1)

        diff2 = (patches - targets).pow(2)
        weighted = diff2 * kernels
        E_per = 0.5 * weighted.sum(dim=(1, 2, 3)) * self._amp_factor()
        return E_per if per_sample else E_per.sum()

    def im2col(self):
        x = self._layer_pre.state
        N = x.shape[0]
        weight, C_out, _, Kh, Kw, _, _, H_out, W_out = self._conv_geometry()

        cols = F.unfold(x, (Kh, Kw), padding=self._P, stride=self._S, dilation=self._D)
        Wflat = weight.view(C_out, -1)
        y = torch.matmul(cols.transpose(1, 2), Wflat.t())
        y = y.transpose(1, 2).reshape(N, C_out, H_out, W_out)
        return y

    def col2im(self):
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._conv_geometry()
        y = self._layer_post.state
        N = y.shape[0]

        L = H_out * W_out
        y_cols = y.reshape(N, C_out, L)
        Wflat = weight.view(C_out, -1)
        cols_pre = torch.matmul(Wflat.t().unsqueeze(0), y_cols)

        x_pre = F.fold(
            cols_pre,
            output_size=(H_in, W_in),
            kernel_size=(Kh, Kw),
            padding=self._P,
            stride=self._S,
            dilation=self._D,
        )
        return x_pre

    def a_im2col(self):
        weight, C_out, C_in, Kh, Kw, _, _, H_out, W_out = self._conv_geometry()
        a_per_ch = weight.view(C_out, -1).sum(dim=1).view(1, C_out, 1, 1)
        return a_per_ch.expand(1, C_out, H_out, W_out)

    def a_col2im(self):
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._conv_geometry()
        y = self._layer_post.state
        y_ones = torch.ones_like(y)
        N = y_ones.shape[0]

        L = H_out * W_out
        y_cols = y_ones.reshape(N, C_out, L)
        Wflat = weight.view(C_out, -1)
        cols_pre = torch.matmul(Wflat.t().unsqueeze(0), y_cols)
        x_pre = F.fold(
            cols_pre,
            output_size=(H_in, W_in),
            kernel_size=(Kh, Kw),
            padding=self._P,
            stride=self._S,
            dilation=self._D,
        )
        return x_pre

    def a_coef_fn(self, layer):
        dictionary = {
            self._layer_pre: self._a_coef_layer_pre,
            self._layer_post: self._a_coef_layer_post,
        }
        return dictionary[layer]

    def b_coef_fn(self, layer):
        dictionary = {
            self._layer_pre: self._b_coef_layer_pre,
            self._layer_post: self._b_coef_layer_post,
        }
        return dictionary[layer]

    def _b_coef_layer_pre(self):
        b_coef = -self.col2im()
        b_coef = b_coef * self._amp_factor() * self._pre_scale()
        return b_coef

    def _b_coef_layer_post(self):
        b_coef = -self.im2col()
        return b_coef * self._amp_factor() * self._pre_scale()

    def _a_coef_layer_pre(self):
        a_map = self.a_col2im()
        a_coef = a_map * self._amp_factor() * self._pre_scale() ** 2
        return 0.5 * a_coef

    def _a_coef_layer_post(self):
        a_map = self.a_im2col()
        return 0.5 * a_map * self._amp_factor()

    def _grad_weight(self):
        weight, C_out, C_in, Kh, Kw, *_ = self._conv_geometry()
        x = self._layer_pre.state * self._pre_scale()
        y = self._layer_post.state.clone()

        cols = F.unfold(x, (Kh, Kw), padding=self._P, stride=self._S, dilation=self._D)
        N, C_out, H_out, W_out = y.shape
        K = C_in * Kh * Kw
        L = H_out * W_out

        patches = cols.transpose(1, 2).unsqueeze(2)
        targets = y.reshape(N, C_out, L).transpose(1, 2).unsqueeze(-1)
        diff2 = (patches - targets).pow(2)
        grad_weight = 0.5 * diff2.sum(dim=1).mean(dim=0)

        return grad_weight.view(C_out, C_in, Kh, Kw) * self._amp_factor()

    def grad_param_fn(self, param):
        dictionary = {self._weight: self._grad_weight}
        return dictionary[param]
    

class BasePoolResistive(QFunction, ABC):
    def __init__(self, layer_pre, layer_post, pool_weight, stride, voltage_amp, current_amp):
        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight = pool_weight
        self._stride = stride
        QFunction.__init__(self, [layer_pre, layer_post], [pool_weight]) ##I am not sure whether to include pool_weight in the parameters
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp
        self._S = stride       
        self._P = 0
        self._D = 1

    def _amp_factor(self):
        return (self._current_amp / self._voltage_amp) ** layer_index(self._layer_pre)

    def _pre_scale(self):
        return 1.0 if layer_index(self._layer_pre) == 0 else self._voltage_amp


    def _pool_geometry(self):
        weight = self._weight.get()
        C_out, C_in, Kh, Kw = weight.shape
        if C_out != C_in:
            raise ValueError("C_out must equal C_in for pooling")
        x = self._layer_pre.state
        _, _, H_in, W_in = x.shape

        H_out = (H_in + 2 * self._P - self._D * (Kh - 1) - 1) // self._S + 1
        W_out = (W_in + 2 * self._P - self._D * (Kw - 1) - 1) // self._S + 1

        return weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out


    @abstractmethod
    def eval(self, per_sample=True):
        pass


    @abstractmethod
    def im2col(self):
        pass


    @abstractmethod
    def col2im(self):
        pass



    @abstractmethod
    def a_im2col(self):
        pass


    @abstractmethod
    def a_col2im(self):
        pass

    def a_coef_fn(self, layer):
        dictionary = {
            self._layer_pre: self._a_coef_layer_pre,
            self._layer_post: self._a_coef_layer_post,
        }
        return dictionary[layer]

    def b_coef_fn(self, layer):
        dictionary = {
            self._layer_pre: self._b_coef_layer_pre,
            self._layer_post: self._b_coef_layer_post,
        }
        return dictionary[layer]

    def _b_coef_layer_pre(self):
        b_coef = -self.col2im()
        b_coef = b_coef * self._amp_factor() * self._pre_scale()
        return b_coef

    def _b_coef_layer_post(self):
        b_coef = -self.im2col()
        return b_coef * self._amp_factor() * self._pre_scale()

    def _a_coef_layer_pre(self):
        a_map = self.a_col2im()
        a_coef = a_map * self._amp_factor() * self._pre_scale() ** 2
        return 0.5 * a_coef

    def _a_coef_layer_post(self):
        a_map = self.a_im2col()
        return 0.5 * a_map * self._amp_factor()

    def _grad_weight(self):
        weight, C_out, C_in, Kh, Kw, *_ = self._pool_geometry()
        return torch.zeros_like(weight)

    def grad_param_fn(self, param):
        return {self._weight: self._grad_weight}.get(param, lambda: torch.zeros(1, device=self._weight.state.device))



class AveragePoolResistive(BasePoolResistive):
    def __init__(self, layer_pre, layer_post, pool_weight, stride, voltage_amp, current_amp):
        BasePoolResistive.__init__(self, layer_pre, layer_post, pool_weight, stride, voltage_amp, current_amp)


    def eval(self, per_sample=True):
        weight, C_out, C_in, Kh, Kw, *_ = self._pool_geometry()

        layer_pre = self._layer_pre.state * self._pre_scale()
        layer_post_scaled = self._layer_post.state


        cols = F.unfold(layer_pre, (Kh, Kw), padding=self._P, stride=self._S, dilation=self._D)
        N, C_out, H_out, W_out = layer_post_scaled.shape
        K = C_in * Kh * Kw
        L = H_out * W_out

        patches = cols.transpose(1, 2).unsqueeze(2)
        kernels = weight.view(1, 1, C_out, K)
        targets = layer_post_scaled.view(N, C_out, L).transpose(1, 2).unsqueeze(-1)

        diff2 = (patches - targets).pow(2)
        weighted = diff2 * kernels
        E_per = 0.5 * weighted.sum(dim=(1, 2, 3)) * self._amp_factor()
        return E_per if per_sample else E_per.sum()

    def im2col(self):
        x = self._layer_pre.state
        N = x.shape[0]
        weight, C_out, _, Kh, Kw, _, _, H_out, W_out = self._pool_geometry()

        cols = F.unfold(x, (Kh, Kw), padding=self._P, stride=self._S, dilation=self._D)
        Wflat = weight.view(C_out, -1)
        y = torch.matmul(cols.transpose(1, 2), Wflat.t())
        y = y.transpose(1, 2).reshape(N, C_out, H_out, W_out)
        return y

    def col2im(self):
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._pool_geometry()
        y = self._layer_post.state
        N = y.shape[0]

        L = H_out * W_out
        y_cols = y.reshape(N, C_out, L)
        Wflat = weight.view(C_out, -1)
        cols_pre = torch.matmul(Wflat.t().unsqueeze(0), y_cols)

        x_pre = F.fold(
            cols_pre,
            output_size=(H_in, W_in),
            kernel_size=(Kh, Kw),
            padding=self._P,
            stride=self._S,
            dilation=self._D,
        )
        return x_pre

    def a_im2col(self):
        weight, C_out, C_in, Kh, Kw, _, _, H_out, W_out = self._pool_geometry()
        a_per_ch = weight.view(C_out, -1).sum(dim=1).view(1, C_out, 1, 1)
        return a_per_ch.expand(1, C_out, H_out, W_out)

    def a_col2im(self):
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._pool_geometry()
        y = self._layer_post.state
        y_ones = torch.ones_like(y)
        N = y_ones.shape[0]

        L = H_out * W_out
        y_cols = y_ones.reshape(N, C_out, L)
        Wflat = weight.view(C_out, -1)
        cols_pre = torch.matmul(Wflat.t().unsqueeze(0), y_cols)
        x_pre = F.fold(
            cols_pre,
            output_size=(H_in, W_in),
            kernel_size=(Kh, Kw),
            padding=self._P,
            stride=self._S,
            dilation=self._D,
        )
        return x_pre

class MaxPoolResistive(BasePoolResistive):
    def __init__(self, layer_pre, layer_post, pool_weight, stride, voltage_amp, current_amp, max_pooling_mode = "abs_pool"):
        BasePoolResistive.__init__(self, layer_pre, layer_post, pool_weight, stride, voltage_amp, current_amp)
        self.pooling_mode = max_pooling_mode #abs_pool or none
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._pool_geometry()
        self.gain = self._weight.get().mean().item() * (Kh * Kw) 
        self.f_idx = None
        self.b_idx = None


    def eval(self, per_sample=True):
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._pool_geometry()

        x = self._layer_pre.state * self._pre_scale()
        y = self._layer_post.state

        N = x.shape[0]

        # Winner selection (consistent with your abs_pool routing)
        if self.pooling_mode == "abs_pool":
            _, idx = F.max_pool2d(
                x.abs(),
                kernel_size=(Kh, Kw),
                stride=self._S,
                padding=self._P,
                dilation=self._D,
                return_indices=True,
            )
            # idx shape: (N, C_in, H_out, W_out) since C_out == C_in for pooling
            x_flat = x.view(N, C_in, -1)
            idx_flat = idx.view(N, C_in, -1)
            x_win = torch.gather(x_flat, 2, idx_flat).view(N, C_in, H_out, W_out)
        else:
            # Standard maxpool (signed)
            x_win = F.max_pool2d(
                x,
                kernel_size=(Kh, Kw),
                stride=self._S,
                padding=self._P,
                dilation=self._D,
            )

        # Effective conductance (broadcast safely)
        g = self.gain
        if not torch.is_tensor(g):
            g = torch.tensor(g, device=x.device, dtype=x.dtype)

        if g.ndim == 0:
            g = g.view(1, 1, 1, 1)              # scalar gain
        elif g.ndim == 1:
            g = g.view(1, C_in, 1, 1)            # per-channel gain

        # Energy: 1/2 * g * (x_win - y)^2
        diff2 = (x_win - y).pow(2)
        E_per = 0.5 * (diff2 * g).sum(dim=(1, 2, 3)) * self._amp_factor()

        return E_per if per_sample else E_per.sum()



    def mymax_pool2d(self, x, kernel_size):
        stride = self._S
        padding = self._P
        dilation = self._D
        N = x.shape[0]
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._pool_geometry()
        if self.pooling_mode == "abs_pool":
            _, idx = F.max_pool2d(abs(x), kernel_size = kernel_size, stride = self._S, padding = self._P, dilation = self._D, return_indices = True) 
            out = torch.gather(x.view(N, C_in, -1), 2, idx.view(N, C_out, -1))
            values = out.view(N, C_out, H_out, W_out)
            return values, idx

        else:
            values, idx = F.max_pool2d(x, kernel_size = kernel_size, stride = self._S, padding = self._P, dilation = self._D, return_indices = True)
            return values, idx

    def mymax_unpool2d(self, x , y, kernel_size):
        stride = self._S
        padding = self._P
        dilation = self._D
        N = y.shape[0]
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._pool_geometry()
        if self.pooling_mode == "abs_pool":
            _, idx = F.max_pool2d(abs(x), kernel_size = kernel_size, stride = self._S, padding = self._P, dilation = self._D, return_indices = True) 
            self.b_idx = idx
            out = torch.gather(x.view(N, C_in, -1), 2, idx.view(N, C_out, -1))
            values = out.view(N, C_out, H_out, W_out)
            x_pre = F.max_unpool2d(input = y, indices = idx, kernel_size = kernel_size, stride=self._S, padding=self._P,output_size=(N, C_out, H_in, W_in))

        else:
            _, idx = F.max_pool2d(x, kernel_size = kernel_size, stride = self._S, padding = self._P, dilation = self._D, return_indices = True) 
            x_pre = F.max_unpool2d(input = y, indices = idx, kernel_size = kernel_size, stride=self._S, padding=self._P, output_size=(N, C_out, H_in, W_in))

        if self.f_idx is not None and self.b_idx is not None:
            diff = self.f_idx - self.b_idx
            mismatches = torch.count_nonzero(diff)
            mismatched_norm = mismatches/diff.numel()
            self.f_idx = None
            self.b_idx = None

        return x_pre

    def im2col(self):
        #maybe I can include the gain here
        x = self._layer_pre.state
        weight, _, _, Kh, Kw, *_ = self._pool_geometry()
        y, idx = self.mymax_pool2d(x, kernel_size=(Kh, Kw))
        self.f_idx = idx
        return y  * self.gain
    def col2im(self):
        #maybe I can include the gain here
        x = self._layer_pre.state
        y = self._layer_post.state
        _, _, _, Kh, Kw, _, _, _, _ = self._pool_geometry()
        x_pre = self.mymax_unpool2d(x, y, kernel_size=(Kh, Kw))
        return x_pre * self.gain

    def a_im2col(self):
        #this makes sense if the weight I consider is alwazs self.gain
        #this is independent of the winners
        weight, C_out, C_in, Kh, Kw, _, _, H_out, W_out = self._pool_geometry()
        a_per_ch = weight.new_ones(1, C_out, H_out, W_out) * self.gain
        return a_per_ch.expand(1, C_out, H_out, W_out)

    def a_col2im(self):
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._pool_geometry()
        x = self._layer_pre.state
        y = self._layer_post.state
        y_ones = torch.ones_like(y) * self.gain
        N = y_ones.shape[0]

        L = H_out * W_out
        y_cols = y_ones.reshape(N, C_out, L)
        x_mask = self.mymax_unpool2d(x, y_ones, kernel_size=(Kh, Kw))   
        return x_mask


    def analytical_grad_layer_fn(self, layer):
            """Returns the gradient of the function wrt the layer, i.e. dE/dz, where z is the layer

            Overrides the default implementation of the class Function

            By assumption, the function E as a function of z is of the form E(z) = a * z^2 + b * z + c.
            So, the gradient can be calculated as dE/dz = 2 a * z + b

            Args:
                layer (Layer): the layer whose gradient we want to compute

            Returns:
                Tensor of shape (batch_size, layer_shape). Type is float32
            """
            a_fn = self.a_coef_fn(layer)
            b_fn = self.b_coef_fn(layer)
            return lambda: 2. * a_fn() * layer.state + b_fn()  # tensor of size (batch_size, layer_shape)
