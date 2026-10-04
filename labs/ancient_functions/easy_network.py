import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
PROJECT_ROOT = Path(__file__).resolve().parent / "energy-based-learning"
LABS_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(LABS_ROOT) not in sys.path:
    sys.path.append(str(LABS_ROOT))

from model.function.cost import SquaredErrorPairedOutputs
from model.function.network import Network
from custom_minimizer import CustomQuadraticMinimizer as QuadraticMinimizer
from model.function.interaction import (
    BiasInteraction,
    DoubleExponentialNonLinearInteraction,
    DoubleQuadraticNonLinearInteraction,
    SumSeparableFunction,
)
from model.resistive.interaction import DenseResistive
from model.resistive.layer import NonlinearResistiveLayer
from model.variable.layer import InputLayer, LinearLayer
from model.variable.parameter import Bias, DenseWeight
from training.core.sgd import AugmentedFunction, Nudging



class TrackingQuadraticMinimizer(QuadraticMinimizer):
    def __init__(self, fn, free_layers, *args, **kwargs):
        self._tracked_fn = fn
        super().__init__(fn, free_layers, *args, **kwargs)
        self.gradient_history_before_sample = {}
        self.gradient_history_after_sample = {}
        self.gradient_history_before = []
        self.gradient_history_after = []

    def _step(self,layer_group):
        pre_activations = [updater.pre_activate() for updater in layer_group]
        for updater, pre_activation in zip(layer_group, pre_activations):
            updater._layer.state = pre_activation

    def step(self, layer_group):
        grads_before = {}
        for updater in layer_group:
            layer = updater._layer
            grad_fn = self._tracked_fn.grad_layer_fn(layer)
            grad = grad_fn().detach()
            grad_norm = float(torch.norm(grad, p=float("inf")).cpu())
            if layer.name not in self.gradient_history_before_sample:
                self.gradient_history_before_sample[layer.name] = []
            self.gradient_history_before_sample[layer.name].append(grad_norm)




        super().step(layer_group)

        grads_after = {}
        for updater in layer_group:
            layer = updater._layer
            grad_fn = self._tracked_fn.grad_layer_fn(layer)
            grad = grad_fn().detach()
            grad_norm = float(torch.norm(grad, p=float("inf")).cpu())
            if layer.name not in self.gradient_history_after_sample:
                self.gradient_history_after_sample[layer.name] = []
            self.gradient_history_after_sample[layer.name].append(grad_norm)

    
    def update_gradients(self):
        self.gradient_history_before.append(self.gradient_history_before_sample)
        self.gradient_history_after.append(self.gradient_history_after_sample)
        self.gradient_history_before_sample = {}
        self.gradient_history_after_sample = {}


class FlexibleResistiveInputLayer(InputLayer):
    """Input layer with a configurable default duplication mode."""

    def __init__(self, shape, gain, batch_size=1, device=None, default_mode="train"):
        super().__init__(shape, batch_size=batch_size, device=device)
        self._gain = gain
        self._default_mode = default_mode

    def set_mode(self, mode: str) -> None:
        if mode not in {"train", "test"}:
            raise ValueError(f"Unknown input mode: {mode}")
        self._default_mode = mode

    def mode(self) -> str:
        return self._default_mode

    def set_input(self, input_values, mode=None):
        chosen_mode = self._default_mode if mode is None else mode
        if chosen_mode == "train":
            self._state = self._gain * torch.cat((input_values, -input_values), 1)
        elif chosen_mode == "test":
            self._state = self._gain * torch.cat((input_values, torch.zeros_like(input_values)), 1)
        else:
            raise ValueError(f"Unknown input mode: {chosen_mode}")


class FlexibleDeepResistiveEnergy(SumSeparableFunction):
    """
    Drop-in replacement for DeepResistiveEnergy using FlexibleResistiveInputLayer.
    """

    def __init__(
        self,
        layer_shapes,
        weight_gains,
        input_gain,
        non_linearity,
        exponential_diode_param,
        quadratic_diode_param,
        voltage_amp,
        current_amp,
        weight_min=None,
        weight_max=None,
        input_mode="train",
    ):
        self._input_amplifier = input_gain
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp
        self._non_linearity = non_linearity

        self._layer_shapes = layer_shapes
        self._weight_gains = weight_gains
        self._weight_min = weight_min
        self._weight_max = weight_max

        input_shape = layer_shapes[0]
        hidden_shapes = layer_shapes[1:-1]
        output_shape = layer_shapes[-1]

        input_layer = FlexibleResistiveInputLayer(
            input_shape, gain=input_gain, device=None, default_mode=input_mode
        )
        hidden_layers = [
            NonlinearResistiveLayer(shape, non_linearity=non_linearity)
            for shape in hidden_shapes
        ]
        output_layer = LinearLayer(output_shape, device=None)
        layers = [input_layer] + hidden_layers + [output_layer]

        biases = [Bias(shape, 0.0, device=None) for shape in layer_shapes]
        bias_interactions = [BiasInteraction(layer, bias) for layer, bias in zip(layers, biases)]

        weights = [
            DenseWeight(
                shape_pre,
                shape_post,
                gain,
                device=None,
                clamp=True,
                clamp_min=weight_min,
                clamp_max=weight_max,
                init_mode="xavier-uniform",
            )
            for shape_pre, shape_post, gain in zip(layer_shapes[:-1], layer_shapes[1:], weight_gains)
        ]
        weight_interactions = [
            DenseResistive(layer_pre, layer_post, weight, self._voltage_amp, self._current_amp)
            for layer_pre, layer_post, weight in zip(layers[:-1], layers[1:], weights)
        ]

        if non_linearity == "perfect_diode":
            non_linear_interaction = []
        elif non_linearity == "double_diode_quadratic":
            non_linear_interaction = [
                DoubleQuadraticNonLinearInteraction(
                    layer, quadratic_diode_param, voltage_amp=self._voltage_amp
                )
                for layer in hidden_layers
            ]
        elif non_linearity == "double_diode_exponential":
            non_linear_interaction = [
                DoubleExponentialNonLinearInteraction(
                    layer,
                    exponential_diode_param,
                    voltage_amp=self._voltage_amp,
                    current_amp=self._current_amp,
                )
                for layer in hidden_layers
            ]
        else:
            raise ValueError(f"Unknown non-linearity: {non_linearity}")

        params = biases[1:] + weights
        interactions = bias_interactions[1:] + weight_interactions + non_linear_interaction

        SumSeparableFunction.__init__(self, layers, params, interactions)

    def set_input_mode(self, mode: str) -> None:
        input_layer = self.layers()[0]
        if isinstance(input_layer, FlexibleResistiveInputLayer):
            input_layer.set_mode(mode)
        else:
            raise TypeError("Input layer is not FlexibleResistiveInputLayer")

    """Create a combined plot of network scores and discrete predictions vs an input feature."""
    if not len(inputs) or not len(scores):
        return None

    base_dir.mkdir(parents=True, exist_ok=True)

    inputs = np.asarray(inputs)
    if inputs.ndim == 1:
        primary_input = inputs
    else:
        primary_input = inputs[:, feature_idx]
    scores = np.asarray(scores)
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    sort_idx = np.argsort(primary_input)
    sorted_inputs = primary_input[sort_idx]
    sorted_scores = scores[sort_idx]
    sorted_predictions = predictions[sort_idx]
    sorted_labels = labels[sort_idx]

    fig, (ax_scores, ax_classes) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for dim in range(sorted_scores.shape[1]):
        ax_scores.plot(sorted_inputs, sorted_scores[:, dim], label=f"score[{dim}]")
    ax_scores.set_ylabel("Score")
    ax_scores.set_title(f"Network scores vs input[{feature_idx}]")
    ax_scores.grid(True, alpha=0.3)
    if sorted_scores.shape[1] <= 5:
        ax_scores.legend(fontsize=7, ncol=sorted_scores.shape[1])

    ax_classes.scatter(sorted_inputs, sorted_predictions, s=10, alpha=0.6, label="prediction")
    ax_classes.scatter(sorted_inputs, sorted_labels, s=10, alpha=0.6, label="label")
    ax_classes.set_xlabel(f"Input[{feature_idx}] value")
    ax_classes.set_ylabel("Class")
    ax_classes.set_title("Predicted vs true class")
    ax_classes.grid(True, alpha=0.3)
    ax_classes.legend(fontsize=7)

    plt.tight_layout()
    path = base_dir / "input_output_relationship.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved input/output plot: {path}")
    return path

def toy_run(num_samples=10, noise=0.1, beta=0, hidden_units=8, input_mode="test"):
    """Run a tiny DRN on the moons dataset and collect energy gradients."""

    project_root = Path(__file__).resolve().parent / "energy-based-learning"
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1

    n_features = 2
    n_classes = 2
    input_shape = (8,)
    hidden_shape = (2,)
    output_shape = ( 4,)
    layer_shapes = [input_shape, hidden_shape, output_shape]
    weight_gains = [0.1, 0.1]
    quadratic_diode_param=  {
        "diode_conductance": 1e-6,
        "v_min": -0.3,
        "v_max": 0.3}
    exponential_diode_param = {
        "I_s": 1e-6,
        "V_t": 0.05,
        "V_off": 0.5}
    non_linearity = "double_diode_exponential"
    mode = "asynchronous"
    energy_fn = FlexibleDeepResistiveEnergy(
        layer_shapes=layer_shapes,
        weight_gains=weight_gains,
        input_gain=1.0,
        non_linearity=non_linearity,
        exponential_diode_param = exponential_diode_param,
        quadratic_diode_param =quadratic_diode_param,
        voltage_amp=1.0,
        current_amp=1.0,
        weight_min=1e-7,
        weight_max=1.0,
        input_mode=input_mode,
    )
    energy_fn.set_device(device)

    network = Network(energy_fn)
    free_layers = network.free_layers()
    output_layer = energy_fn.layers()[-1]

    cost_fn = SquaredErrorPairedOutputs(output_layer, num_classes=n_classes)
    nudging_wrapper = Nudging(cost_fn)
    augmented_fn = AugmentedFunction(energy_fn, cost_fn)

    minimizer = TrackingQuadraticMinimizer(
        augmented_fn,
        free_layers,
        num_iterations=15,
        mode=mode,
        non_linearity=non_linearity,
        exponential_diode_param=exponential_diode_param,
        quadratic_diode_param=quadratic_diode_param,
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
    )

    _, labels = make_moons(n_samples=num_samples, noise=noise, random_state=42)
    num_inputs = input_shape[0]//2
    if num_inputs % 2 != 0:
        raise ValueError("toy_run expects an even number of raw input features")
    linspace_inputs = np.linspace(-10, 10, num_samples, dtype=np.float32)
    custom_data = np.zeros((num_samples, num_inputs), dtype=np.float32)
    custom_data[:, : num_inputs // 2] = linspace_inputs[:, None]
    x = torch.tensor(custom_data, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    augmented_fn.nudging = 0
    plots_dir = Path(__file__).resolve().parent / "plots" / "easy_network"
    layer_response_dir = plots_dir / "layer_responses"
    input_output_dir = plots_dir / "input_output"
    gradient_dir = plots_dir / "gradients"

    minimizer.gradient_history_before = []
    minimizer.gradient_history_after = []
    minimizer.gradient_history_before_sample = {}
    minimizer.gradient_history_after_sample = {}

    total_start = time.perf_counter()
    batch_times = []
    equilibrium_times = []

    results = []
    sample_inputs = []
    sample_labels = []
    sample_scores = []
    sample_predictions = []
    sample_output_states = []
    tracked_layers = energy_fn.layers()
    layer_state_history = {layer.name: [] for layer in tracked_layers}

    for step, (x_batch, y_batch) in enumerate(data_loader):
        batch_start = time.perf_counter()
        labels = y_batch.to(device)
        batch_inputs_cpu = x_batch.detach().cpu()
        batch_labels_cpu = y_batch.detach().cpu()

        network.set_input(x_batch.to(device), reset=True)
        cost_fn.set_target(labels)
        #nudging_wrapper.nudging = beta
        equilibrium_start = time.perf_counter()
        minimizer.compute_equilibrium()
        equilibrium_duration = time.perf_counter() - equilibrium_start
        minimizer.update_gradients()
        equilibrium_times.append(equilibrium_duration)

        loss_value = cost_fn.eval().mean().item()
        scores = cost_fn._scores(output_layer.state.detach())
        predictions = scores.argmax(dim=1)
        acc = (predictions == labels).float().mean().item()

        scores_cpu = scores.detach().cpu()
        predictions_cpu = predictions.detach().cpu()
        output_states_cpu = output_layer.state.detach().cpu()
        layer_states_batch = {
            layer.name: layer.state.detach().cpu() for layer in tracked_layers
        }

        batch_size_actual = batch_inputs_cpu.shape[0]
        for sample_idx in range(batch_size_actual):
            input_sample = batch_inputs_cpu[sample_idx].numpy()
            label_sample = int(batch_labels_cpu[sample_idx].item())
            score_sample = scores_cpu[sample_idx].numpy()
            prediction_sample = int(predictions_cpu[sample_idx].item())
            output_state_sample = output_states_cpu[sample_idx].numpy()

            sample_inputs.append(input_sample)
            sample_labels.append(label_sample)
            sample_scores.append(score_sample)
            sample_predictions.append(prediction_sample)
            sample_output_states.append(output_state_sample)

            for layer_name, layer_state_tensor in layer_states_batch.items():
                layer_state_history[layer_name].append(
                    layer_state_tensor[sample_idx].reshape(-1).numpy()
                )

            results.append(
                {
                    "sample_index": len(sample_inputs) - 1,
                    "batch": step,
                    "loss": loss_value,
                    "accuracy": acc,
                    "input": input_sample.tolist(),
                    "prediction": prediction_sample,
                    "label": label_sample,
                }
            )

        batch_duration = time.perf_counter() - batch_start
        batch_times.append(batch_duration)

    mean_grad_before = aggregate_gradient_history(minimizer.gradient_history_before)
    mean_grad_after = aggregate_gradient_history(minimizer.gradient_history_after)

    gradient_plot_before = plot_average_gradient_history(
        mean_grad_before,
        "Mean gradient norms before weight updates",
        gradient_dir,
        "gradients_before",
    )
    gradient_plot_after = plot_average_gradient_history(
        mean_grad_after,
        "Mean gradient norms after weight updates",
        gradient_dir,
        "gradients_after",
    )

    input_output_plot_path = plot_input_output(
        sample_inputs,
        sample_scores,
        sample_predictions,
        sample_labels,
        input_output_dir,
        feature_idx=0,
    )

    layer_response_paths = {}
    for layer_name, history in layer_state_history.items():
        path = plot_layer_state_over_inputs(
            layer_name, history, sample_inputs, layer_response_dir, feature_idx=0
        )
        if path is not None:
            layer_response_paths[layer_name] = path

    layer_state_arrays = {
        layer_name: np.stack(history, axis=0)
        for layer_name, history in layer_state_history.items()
        if history
    }

    total_duration = time.perf_counter() - total_start
    avg_batch_time = float(np.mean(batch_times)) if batch_times else 0.0
    avg_equilibrium_time = float(np.mean(equilibrium_times)) if equilibrium_times else 0.0
    max_batch_time = float(np.max(batch_times)) if batch_times else 0.0
    max_equilibrium_time = float(np.max(equilibrium_times)) if equilibrium_times else 0.0

    timing_stats = {
        "total_seconds": total_duration,
        "avg_batch_seconds": avg_batch_time,
        "avg_equilibrium_seconds": avg_equilibrium_time,
        "max_batch_seconds": max_batch_time,
        "max_equilibrium_seconds": max_equilibrium_time,
        "num_batches": len(batch_times),
    }

    summary = {
        "metrics": results,
        "inputs": np.asarray(sample_inputs),
        "labels": np.asarray(sample_labels),
        "scores": np.asarray(sample_scores),
        "predictions": np.asarray(sample_predictions),
        "output_states": np.asarray(sample_output_states),
        "layer_states": layer_state_arrays,
        "input_output_plot": str(input_output_plot_path) if input_output_plot_path else None,
        "layer_response_plots": {k: str(v) for k, v in layer_response_paths.items()},
        "mean_gradients_before": mean_grad_before,
        "mean_gradients_after": mean_grad_after,
        "mean_gradient_plot_before": str(gradient_plot_before) if gradient_plot_before else None,
        "mean_gradient_plot_after": str(gradient_plot_after) if gradient_plot_after else None,
        "timing_stats": timing_stats,
    }

    return summary



if __name__ == "__main__":
    summary = toy_run()
    metrics = summary["metrics"]
    for sample_summary in metrics[:5]:
        print(
            f"sample={sample_summary['sample_index']} "
            f"batch={sample_summary['batch']} "
            f"loss={sample_summary['loss']:.4f} "
            f"acc={sample_summary['accuracy']:.3f} "
            f"prediction={sample_summary['prediction']} "
            f"label={sample_summary['label']} "
        )
    if summary["input_output_plot"]:
        print(f"Input/output plot saved to {summary['input_output_plot']}")
    if summary["mean_gradient_plot_before"]:
        print(f"Mean gradient (before updates) plot saved to {summary['mean_gradient_plot_before']}")
    if summary["mean_gradient_plot_after"]:
        print(f"Mean gradient (after updates) plot saved to {summary['mean_gradient_plot_after']}")
    timing_stats = summary.get("timing_stats", {})
    if timing_stats:
        print(
            "Timing:"
            f" total={timing_stats['total_seconds']:.2f}s;"
            f" avg_batch={timing_stats['avg_batch_seconds']:.3f}s;"
            f" avg_equilibrium={timing_stats['avg_equilibrium_seconds']:.3f}s;"
            f" batches={timing_stats['num_batches']}"
        )
