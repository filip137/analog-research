"""Per-batch statistics registered on lab runners and recorded by the monitor."""

from abc import ABC, abstractmethod
import torch



class Statistic(ABC):
    """Abstract class for statistics

    Attributes
    ----------

    Methods
    -------
    do_measurement()
        Do a measurement and adjusts the value of the statistic accordingly
    get()
        Returns the current value of the statistic
    reset()
        Resets the statistic to zero
    """

    @abstractmethod
    def do_measurement(self):
        """Do a measurement and adjusts the value of the statistic accordingly"""
        pass

    @abstractmethod
    def get(self):
        """Returns the current value of the statistic"""
        pass

    @abstractmethod
    def reset(self):
        """Resets the statistic to zero"""
        pass


class Counter(Statistic):
    """Class to count how many examples of the dataset have been processed

    Attributes
    ----------
    _num_examples (int): the number of examples processed so far
    _dataset_size (int): the number of examples in the dataset

    Methods
    -------
    _count_fn()
        Counts the number of examples in the current mini-batch
    """

    display = True
    name = None
    option = None
    display_name = None

    def __init__(self, network, dataset_size):
        """Initializes an instance of Counter

        Args:
            network (SumSeparableFunction): the network where we count the number of examples processed
            dataset_size (int): the number of examples in the dataset
        """

        self._network = network
        self._num_examples = 0

        self._dataset_size = dataset_size

    def _count_fn(self):
        """Count function"""
        return self._network.layers()[0].state.shape[0]

    def do_measurement(self):
        """Measures the statistics and adds it to the sum"""
        self._num_examples += self._count_fn()

    def get(self):
        """Returns the number of examples processed so far"""
        return self._num_examples

    def reset(self):
        """Resets the number of examples processed to zero"""
        self._num_examples = 0

    def __str__(self):
        string = 'Example {:5d}/{}'.format(self._num_examples, self._dataset_size)
        return string
        

class MeanStat(Statistic, ABC):
    """
    Abstract class used to accumulate statistics over the mini-batches to get statistics over the entire dataset.

    Attributes
    ----------
    name (str): the name of the statistics
    sum (float32): cumulative sum of the statistics
    num_steps (int): number of times the function "do_meaurement" has been called.
    percentage (bool): whether the statistics is played in percentage or not
    string (str): the string to be formatted.

    Methods
    -------
    do_measurement()
        Measures the statistics and adds it to the sum
    get()
        Returns the mean of the statistics
    reset()
        Resets sum and num_steps to zero
    """

    def __init__(self, display_name=None, name=None, precision=3, percentage=False, display=False):
        """
        Initializes an instance of MeanStat.
        
        Args:
            name (str): the name of the statistics
            precision (int, optional): the number of decimals to display. Default: 3.
            percentage (bool, optional): whether the statistics is displayed in percentage or not. Default: False.
        """

        self.display_name = display_name

        self.name = name

        self._sum = 0.
        self._num_examples = 0

        self._display_string = display_name + ' = {:.' + str(precision) + 'f}'
        if percentage: self._display_string += '%'

        self._percentage = percentage

        self.display = display

    @abstractmethod
    def _measure_fn(self):
        """function that measures the statistics"""
        pass

    def do_measurement(self):
        """Measures the statistics and adds it to the sum"""
        amount = self._measure_fn()
        self._sum += amount.sum().item()
        self._num_examples += amount.numel()

    def get(self):
        """Returns the mean statistics"""
        mean = self._sum / self._num_examples
        if self._percentage: mean *= 100.
        return mean

    def reset(self):
        """Resets the variables to zero"""
        self._sum = 0.
        self._num_examples = 0

    def __str__(self):
        mean = self.get()
        return self._display_string.format(mean)


class EnergyStat(MeanStat):
    """
    Class used to measure the mean value of the energy function (when the network is at equilibrium) over the dataset.
    """

    option = None

    def __init__(self, network):
        """Initializes an instance of EnergyStat

        Args:
            network (SumSeparableFunction): the network whose energy function is measured at equilibrium
        """

        self._network = network
        display_name = 'Energy'
        name = 'Energy'
        precision = 2
        percentage = False
        display = True

        MeanStat.__init__(self, display_name, name, precision, percentage, display)

    def _measure_fn(self):
        """Energy function"""
        return self._network.eval()


class CostStat(MeanStat):
    """
    Class used to measure the mean cost (when the network is at equilibrium, i.e. the `loss') over the dataset.
    """

    option = None

    def __init__(self, cost_fn):
        """Initializes an instance of CostStat

        Args:
            cost_fn (CostFunction): the cost function whose value we measure
        """

        self._cost_fn = cost_fn
        display_name = 'Cost'
        name = 'Cost'
        precision = 5
        percentage = False
        display = True

        MeanStat.__init__(self, display_name, name, precision, percentage, display)

    def _measure_fn(self):
        """Cost function"""
        return self._cost_fn.eval()


class ErrorStat(MeanStat):
    """
    Class used to measure the mean error rate over the dataset.
    """

    def __init__(self, cost_fn):
        """Initializes an instance of ErrorStat

        Args:
            cost_fn (CostFunction): the cost function associated to the error rate
        """

        self.option = None

        self._cost_fn = cost_fn
        display_name = 'Error'
        name = 'Error'
        precision = 3
        percentage = True
        display = True

        MeanStat.__init__(self, display_name, name, precision, percentage, display)

    def _measure_fn(self):
        """Error function"""
        return self._cost_fn.error_fn().type(torch.float)


class TopFiveErrorStat(MeanStat):
    """
    Class used to measure the mean top-5 error rate over the dataset.
    """

    def __init__(self, cost_fn):
        """Initializes an instance of TopFiveErrorStat

        Args:
            cost_fn (CostFunction): the cost function associated to the top-5 error rate
        """

        self.option = None

        self._cost_fn = cost_fn
        display_name = 'Top5Error'
        name = 'Top5Error'
        precision = 3
        percentage = True
        display = True

        MeanStat.__init__(self, display_name, name, precision, percentage, display)

    def _measure_fn(self):
        """Top-5 error function"""
        return self._cost_fn.top_five_error_fn().type(torch.float)


class SaturationStat(MeanStat):
    """
    Class used to measure the mean saturation over the dataset.

    The `saturation' is the fraction of individual variables that are cliped to the min or the max of the variable's state interval
    """

    def __init__(self, variable, non_linearity, v_min=None, v_max=None):
        """Creates an instance of SaturationStat

        Args:
            variable (Variable): the variable whose saturation we want to track
            non_linearity (str): the type of non-linearity ('perfect_diode', 'lpw_diode',
                'hard_sigmoid', 'double_diode_quadratic', 'double_diode_exponential', 'linear')
            v_min (float, optional): minimum voltage for hard_sigmoid case. Required for hard_sigmoid.
            v_max (float, optional): maximum voltage for hard_sigmoid case. Required for hard_sigmoid.
        """

        diode_like = {'hard_sigmoid', 'double_diode_quadratic', 'double_diode_exponential'}
        allowed = {'perfect_diode', 'lpw_diode', 'linear', 'single_diode_exponential'} | diode_like
        if non_linearity not in allowed:
            raise ValueError(
                "non_linearity must be one of {}, got '{}'".format(sorted(allowed), non_linearity)
            )

        if non_linearity in diode_like:
            if v_min is None or v_max is None:
                raise ValueError("v_min and v_max must be provided for {} non_linearity".format(non_linearity))
            if v_min >= v_max:
                raise ValueError(f"v_min ({v_min}) must be less than v_max ({v_max})")

        self.option = variable.name
        self._variable = variable
        self._non_linearity = non_linearity
        self._v_min = v_min
        self._v_max = v_max

        display_name = 'Saturation_{}'.format(variable.name)
        name = 'Saturation'
        precision = 1
        percentage = True
        display = False

        MeanStat.__init__(self, display_name, name, precision, percentage, display)

    def _measure_fn(self):
        """Saturation function - adapts based on non-linearity type"""
        diode_like = {'hard_sigmoid', 'double_diode_quadratic', 'double_diode_exponential'}
        if self._non_linearity in diode_like:
            # For diode-like: saturation = values outside [v_min, v_max]
            below_min = self._variable.state < self._v_min
            above_max = self._variable.state > self._v_max
            return (below_min | above_max).type(torch.float)
        else:
            # For perfect_diode and lpw_diode: saturation = values exactly at zero
            return (self._variable.state == 0.).type(torch.float)


class NormStat(MeanStat):
    """
    Class used to measure the mean norm of a layer over the dataset.
    """

    def __init__(self, variable):
        """Creates and instance of NormStat

        Args:
            variable (Variable): the variable whose norm we want to track
        """

        self.option = variable.name
        self._variable = variable

        display_name = 'Norm_{}'.format(variable.name)
        name = 'Norm'
        precision = 3
        percentage = False
        display = False

        MeanStat.__init__(self, display_name, name, precision, percentage, display)

    def _measure_fn(self):
        """Norm function"""
        return torch.abs(self._variable.state)


class GradientStat(MeanStat):
    """
    Class used to measure the gradient of a given variable.
    """

    def __init__(self, variable):
        """Creates and instance of GradientStat

        Args:
            variable (Variable): the variable whose gradient we want to monitor
        """

        self.option = variable.name
        self._variable = variable

        display_name = 'Gradient_{}'.format(variable.name)
        name = 'Gradient'
        precision = 5
        percentage = False
        display = False

        MeanStat.__init__(self, display_name, name, precision, percentage, display)

    def _measure_fn(self):
        """Gradient function"""
        return torch.abs(self._variable.state.grad)


class WeightRowSumStat(MeanStat):
    """
    Class used to measure the sum of rows in weight matrices, normalized by batch size.
    """

    def __init__(self, variable):
        """Creates an instance of WeightRowSumStat

        Args:
            variable (Variable): the variable (weight matrix) whose row sums we want to track
        """

        self.option = variable.name
        self._variable = variable

        display_name = 'WeightRowSum_{}'.format(variable.name)
        name = 'WeightRowSum'
        precision = 3
        percentage = False
        display = False

        MeanStat.__init__(self, display_name, name, precision, percentage, display)

    def _measure_fn(self):
        """Row sum function - calculates sum of rows normalized by batch size"""
        weight = self._variable.state
        
        # Get batch size from the first dimension
        batch_size = weight.shape[0] if len(weight.shape) > 1 else 1
        
        # Sum along rows (dim=1) and normalize by batch size
        row_sums = weight.sum(dim=1)
        normalized_row_sums = row_sums / batch_size
        
        return normalized_row_sums.mean()


class WeightColumnSumStat(MeanStat):
    """
    Class used to measure the sum of columns in weight matrices, normalized by batch size.
    """

    def __init__(self, variable):
        """Creates an instance of WeightColumnSumStat

        Args:
            variable (Variable): the variable (weight matrix) whose column sums we want to track
        """

        self.option = variable.name
        self._variable = variable

        display_name = 'WeightColumnSum_{}'.format(variable.name)
        name = 'WeightColumnSum'
        precision = 3
        percentage = False
        display = False

        MeanStat.__init__(self, display_name, name, precision, percentage, display)

    def _measure_fn(self):
        """Column sum function - calculates sum of columns normalized by batch size"""
        weight = self._variable.state
        
        # Get batch size from the first dimension
        batch_size = weight.shape[0] if len(weight.shape) > 1 else 1
        
        # Sum along columns (dim=0) and normalize by batch size
        column_sums = weight.sum(dim=0)
        normalized_column_sums = column_sums / batch_size
        
        return normalized_column_sums.mean()


class WeightDistributionStat(MeanStat):
    """Statistic to measure various weight distribution properties
    
    Attributes
    ----------
    _variable (Variable): the weight variable to measure
    _stat_type (str): the type of statistic to compute ('mean', 'std', 'min', 'max', 'median', 'abs_mean', 'abs_std')
    """
    
    def __init__(self, variable, stat_type='mean'):
        """Creates an instance of WeightDistributionStat
        
        Args:
            variable (Variable): the weight variable to measure
            stat_type (str): the type of statistic ('mean', 'std', 'min', 'max', 'median', 'abs_mean', 'abs_std')
        """
        self.option = variable.name
        self._variable = variable
        self._stat_type = stat_type
        display_name = f'WeightDist_{stat_type}_{variable.name}'
        MeanStat.__init__(self, display_name, 'WeightDist', 3, False, False)
    
    def _measure_fn(self):
        """Compute the specified weight distribution statistic"""
        weight = self._variable.state
        
        if self._stat_type == 'mean':
            return weight.mean()
        elif self._stat_type == 'std':
            return weight.std()
        elif self._stat_type == 'min':
            return weight.min()
        elif self._stat_type == 'max':
            return weight.max()
        elif self._stat_type == 'median':
            return weight.median()
        elif self._stat_type == 'abs_mean':
            return torch.abs(weight).mean()
        elif self._stat_type == 'abs_std':
            return torch.abs(weight).std()
        else:
            return weight.mean()


def add_standard_statistics(runner, energy_fn, cost_fn, *, training=False,
                            non_linearity=None, v_min=None, v_max=None):
    """Register the lab's standard metrics once, during experiment setup.

    Names and units match the historical Monitor series. The monitor only
    records these statistics; it never constructs a second set.
    """
    non_linearity = non_linearity or getattr(energy_fn, "_non_linearity", "perfect_diode")
    if non_linearity in {"hard_sigmoid", "double_diode_quadratic", "double_diode_exponential"}:
        diode_params = getattr(energy_fn, "_quadratic_diode_param", {}) or {}
        v_min = diode_params.get("v_min") if v_min is None else v_min
        v_max = diode_params.get("v_max") if v_max is None else v_max
        if v_min is None or v_max is None:
            raise ValueError(f"v_min and v_max must be provided for {non_linearity} saturation monitoring")
    free = [
        Counter(energy_fn, runner.dataset_size()), EnergyStat(energy_fn),
        CostStat(cost_fn), ErrorStat(cost_fn), TopFiveErrorStat(cost_fn),
    ]
    free += [NormStat(layer) for layer in energy_fn.layers()]
    free += [SaturationStat(layer, non_linearity, v_min, v_max) for layer in energy_fn.layers()]
    for statistic in free:
        runner.add_statistic(statistic, phase="free")
    if training:
        gradients = [GradientStat(parameter) for parameter in energy_fn.params()]
        weights = [parameter for parameter in energy_fn.params() if "Weight" in parameter.name]
        gradients += [WeightRowSumStat(parameter) for parameter in weights]
        gradients += [WeightColumnSumStat(parameter) for parameter in weights]
        gradients += [
            WeightDistributionStat(parameter, kind)
            for kind in ("mean", "std", "min", "max", "abs_mean", "abs_std")
            for parameter in weights
        ]
        for statistic in gradients:
            runner.add_statistic(statistic, phase="gradients")
