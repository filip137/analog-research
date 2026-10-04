"""Digital SGD adapter and update-backend construction.

The core engine consumes an optimizer's step() interface without importing
specific update backends or monitoring infrastructure.
"""

import torch

from model.variable.parameter import PoolWeight


class SGDOptimizer(torch.optim.SGD):

    def __init__(self, energy_fn, cost_fn, learning_rates, momentum=0, weight_decay=0):
        """Creates an instance of the SGD Optimizer.

        Args:
            energy_fn (SumSeparableFunction): the energy function whose parameters we optimize
            cost_fn (CostFunction): the cost function whose parameters we optimize
            learning_rates (list of float): the list of learning rates for the energy parameters and cost parameters
            momentum (float, optional): the momentum. Default: 0.0
            weight_decay (float, optional): the weight decay. Default: 0.0
        """

        self._learning_rates = learning_rates
        self._momentum = momentum
        self._weight_decay = weight_decay

        # Filter out PoolWeight params (kept frozen)
        params = [p for p in (energy_fn.params() + cost_fn.params()) if not isinstance(p, PoolWeight)]
        if len(learning_rates) != len(params):
            raise ValueError(
                f"learning_rates length ({len(learning_rates)}) does not match parameter count ({len(params)} after filtering PoolWeight)"
            )
        params = [{"params": param.state, "lr": lr} for param, lr in zip(params, learning_rates)]
        torch.optim.SGD.__init__(self, params, lr=0.1, momentum=momentum, weight_decay=weight_decay)

    def __str__(self):
        return 'SGD -- initial learning rates = {}, momentum={}, weight_decay={}'.format(self._learning_rates, self._momentum, self._weight_decay)


def build_optimizer(
    energy_fn,
    cost_fn,
    learning_rates,
    *,
    update_pipeline=None,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
):
    """Build the existing SGD optimizer or the optional Tiki-Taka optimizer."""

    from training.tiki_taka import TikiTakaOptimizer, parse_update_pipeline

    config = parse_update_pipeline(update_pipeline)
    if config is None:
        return SGDOptimizer(
            energy_fn,
            cost_fn,
            learning_rates,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    return TikiTakaOptimizer(
        energy_fn,
        cost_fn,
        learning_rates,
        config=config,
        momentum=momentum,
        weight_decay=weight_decay,
    )
