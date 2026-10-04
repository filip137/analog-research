from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Optional

import torch

from training.epoch import Trainer, Evaluator
from training.engine import EvaluationComponents, FreePhaseEvent
from training.diagnostics import FiniteGradientGuard, GradientUpdateObserver, LayerMeasurements
from training.statistics import add_standard_statistics


@dataclass
class MnistParts:
    energy_fn: object
    network: object
    cost_fn: torch.nn.Module
    train_loader: object
    test_loader: object
    free_layers: list
    minimizer_mode: str
    model_cfg: dict
    training_cfg: dict
    # Optional training wiring:
    # - `learning_rates` should align with `SGDOptimizer`'s parameter list (after PoolWeight filtering).
    # - `scheduler` is created by training utilities (e.g. `track_training_statistics`), not during setup.
    learning_rates: Optional[list]
    scheduler: Optional[object]
    voltage_amp: float = 1.0
    current_amp: float = 1.0
    weights_path: Optional[str] = None
    num_iterations: Optional[int] = None
    energy_minimizer_cfg: Optional[dict] = None


def build_evaluator(network, cost_fn, dataloader, energy_minimizer, model_cfg, record_statistics=()):
    """Construct an engine-backed evaluator and register its statistics once."""
    evaluator = CustomEvaluator(
        EvaluationComponents(network, cost_fn, energy_minimizer),
        dataloader,
        reset_input=True,
        record_statistics=record_statistics,
    )
    energy_source = getattr(network, "_function", None) or network
    quad_params = model_cfg.get("quadratic_diode_param", {})
    add_standard_statistics(
        evaluator, energy_source, cost_fn,
        non_linearity=model_cfg["non_linearity"],
        v_min=quad_params.get("v_min"), v_max=quad_params.get("v_max"),
    )
    return evaluator


class _LayerRecording:
    """Expose lab result views backed by an optional engine observer."""

    @property
    def layer_states(self):
        return self._measurements.layer_states

    @property
    def res_currents(self):
        return self._measurements.res_currents

    def _recording_handler(self, record_statistics):
        requested = tuple(record_statistics or ())
        self._measurements = LayerMeasurements(
            store_states="store_states" in requested,
            residual_currents="calc_residual_current" in requested,
        )
        return self._measurements


def _print_interaction_gradients(event):
    """Keep the lab's opt-in circuit diagnostics outside the numerical loop."""
    if not isinstance(event, FreePhaseEvent):
        return
    function = event.components.network._function
    for layer in function.layers():
        contributions = [
            (index, type(interaction).__name__, interaction.grad_layer_fn(layer)())
            for index, interaction in enumerate(function._interactions)
            if layer in interaction.layers()
        ]
        if not contributions:
            continue
        total = sum(gradient for _, _, gradient in contributions)
        print(f"[{layer.name}] total ||grad||={torch.norm(total):.4e}")
        for index, name, gradient in contributions:
            print(f"  {index}:{name} ||grad||={torch.norm(gradient):.4e}")


class CustomTrainer(_LayerRecording, Trainer):
    """Configure lab observers while sharing the core training loop."""

    def __init__(self, components, dataloader, *, reset_input, record_statistics=()):
        handlers = [FiniteGradientGuard(), self._recording_handler(record_statistics)]
        if os.environ.get("DRN_DEBUG_DIODE"):
            handlers.extend((_print_interaction_gradients, GradientUpdateObserver(verbose=True)))
        super().__init__(components, dataloader, reset_input=reset_input,
                         event_handlers=handlers)


class CustomEvaluator(_LayerRecording, Evaluator):
    """Configure lab observers while sharing the core evaluation loop."""

    def __init__(self, components, dataloader, *, reset_input, record_statistics=()):
        super().__init__(
            components, dataloader, reset_input=reset_input,
            event_handlers=(self._recording_handler(record_statistics),),
        )



def export_pt_to_npz(pt_path, npz_path: Optional[Path] = None, param_names=None):
    """Convert a saved model.pt (list of tensors) into a NumPy .npz bundle."""
    import numpy as np

    pt_path = Path(pt_path)
    tensors = torch.load(pt_path, map_location="cpu")
    if not isinstance(tensors, (list, tuple)):
        raise ValueError(f"Expected list/tuple of tensors in {pt_path}, got {type(tensors)}")

    arrays = {}
    for i, tensor in enumerate(tensors):
        name = param_names[i] if param_names and i < len(param_names) else f"param_{i}"
        arrays[name] = tensor.detach().cpu().numpy()

    target = pt_path.with_suffix(".npz") if npz_path is None else Path(npz_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **arrays)
    return target


def flatten_weights_and_inputs(
    weights_npz_path,
    inputs_npz_path,
    output_dir,
    *,
    input_layer: str = "Layer_0",
    shapes_path: Optional[Path] = None,
):
    """Flatten model weights and the first input layer, saving new .npz files."""
    import numpy as np

    weights_npz_path = Path(weights_npz_path)
    inputs_npz_path = Path(inputs_npz_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shapes = {"weights": {}, "inputs": {}}

    with np.load(weights_npz_path) as weights_data:
        flat_weights = {}
        for name, array in weights_data.items():
            if array.ndim > 2:
                flat_array = array.reshape(-1, array.shape[-1])
            else:
                flat_array = array
            flat_weights[name] = flat_array
            shapes["weights"][name] = {
                "original": list(array.shape),
                "flat": list(flat_array.shape),
            }
    with np.load(inputs_npz_path) as inputs_data:
        flat_inputs = {}
        if input_layer not in inputs_data.files:
            input_layer = inputs_data.files[0] if inputs_data.files else input_layer
        for name, array in inputs_data.items():
            if name == input_layer and array.ndim >= 2:
                flat_array = array.reshape(array.shape[0], -1)
            else:
                flat_array = array
            flat_inputs[name] = flat_array
            shapes["inputs"][name] = {
                "original": list(array.shape),
                "flat": list(flat_array.shape),
            }

    weights_out = output_dir / f"{weights_npz_path.stem}_flat.npz"
    inputs_out = output_dir / f"{inputs_npz_path.stem}_flat.npz"
    np.savez(weights_out, **flat_weights)
    np.savez(inputs_out, **flat_inputs)
    if shapes_path is None:
        shapes_path = output_dir / "flattened_shapes.json"
    shapes_path.write_text(json.dumps(shapes, indent=2))
    return weights_out, inputs_out


# ---------- Plotting helpers ----------

def plot_layer_series(x_axis, layer_data, output_dir, prefix, ylabel):
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    for layer_name, tensor in layer_data.items():
        plt.figure(figsize=(8, 4))
        max_units = min(tensor.shape[1], 8)
        for idx in range(max_units):
            plt.plot(x_axis, tensor[:, idx], label=f"{layer_name}[{idx}]")
        plt.xlabel("Input value")
        plt.ylabel(ylabel)
        plt.title(f"{prefix.capitalize()} for {layer_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"{prefix}_{layer_name}.png")
        plt.close()


def plot_layer_extrema(x_axis, max_data, min_data, output_dir):
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    for layer_name in max_data:
        plt.figure(figsize=(8, 4))
        plt.plot(x_axis, max_data[layer_name], label="max")
        plt.plot(x_axis, min_data[layer_name], label="min")
        plt.xlabel("Input value")
        plt.ylabel("State extrema")
        plt.title(f"State extrema for {layer_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"extrema_{layer_name}.png")
        plt.close()


def save_beta_summary(beta_summary, run_dir: Path):
    import matplotlib.pyplot as plt

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "beta_summary.json").write_text(json.dumps(beta_summary, indent=2))
    layers = beta_summary.get("layers", [])
    means = beta_summary.get("per_layer_mean", [])
    medians = beta_summary.get("per_layer_median", [])
    if not layers or not means:
        return
    x = range(len(layers))
    plt.figure(figsize=(10, 5))
    plt.bar(x, means, width=0.4, label="mean", align="center")
    if medians:
        plt.bar([i + 0.4 for i in x], medians, width=0.4, label="median", align="center")
    plt.xticks([i + 0.2 for i in x], layers, rotation=45, ha="right")
    plt.ylabel("ratio (avg displacement / free magnitude)")
    plt.title("Beta-size per layer")
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(run_dir / "beta_summary.png")
    plt.close()


def plot_beta_displacements(beta_results, output_path: Path):
    import matplotlib.pyplot as plt

    if not beta_results:
        return
    layer_sets = [set(entry["beta_summary"].get("layers", [])) for entry in beta_results]
    all_layers = sorted(set().union(*layer_sets))
    if not all_layers:
        return
    x = list(range(len(all_layers)))

    plt.figure(figsize=(10, 5))
    for entry in beta_results:
        summary = entry["beta_summary"]
        layers = summary.get("layers", [])
        means = summary.get("per_layer_mean", [])
        if not layers or not means:
            continue
        mean_map = dict(zip(layers, means))
        aligned = [mean_map.get(name, float("nan")) for name in all_layers]
        label = f"cur={entry['current_amp']}, volt={entry['voltage_amp']}, iters={entry['num_iterations']}"
        plt.plot(x, aligned, marker="o", label=label)

    plt.xticks(x, all_layers, rotation=45, ha="right")
    plt.yscale("log")
    plt.ylabel("Normalized displacement between free and nudged phases")
    plt.xlabel("Layer")
    plt.title("Normalized displacement between free and nudged phases")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_saturation(sat_results, output_path: Path):
    import matplotlib.pyplot as plt

    if not sat_results:
        return
    layer_sets = [set(entry["layers"]) for entry in sat_results]
    all_layers = sorted(set().union(*layer_sets))
    if not all_layers:
        return
    x = list(range(len(all_layers)))

    plt.figure(figsize=(10, 5))
    for entry in sat_results:
        mean_map = dict(zip(entry["layers"], entry["means"]))
        aligned = [mean_map.get(name, float("nan")) for name in all_layers]
        label = f"cur={entry['current_amp']}, volt={entry['voltage_amp']}, iters={entry['num_iterations']}"
        plt.plot(x, aligned, marker="o", label=label)

    plt.ylabel("saturation (mean)")
    plt.xlabel("Layer")
    plt.title("Saturation per layer")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_saturation_from_sweep_results(
    sweep_results_path: Path,
    output_path: Path,
    *,
    normalized_layer_count: int = 4,
    title: str = "Saturation vs normalized layer index",
    voltage_alias: str = "v",
    current_alias: str = "c",
    iterations_alias: str = "iters",
):
    """
    Plot saturation curves from `sweep_results.json` while normalizing layer ids per run.

    Each run may expose saturation keys like Saturation_Layer_4..7, 8..11, etc.
    This function remaps each run's layer ids to a local 0..N-1 index before plotting.
    """
    import matplotlib.pyplot as plt
    import re

    sweep_results_path = Path(sweep_results_path)
    output_path = Path(output_path)
    results = json.loads(sweep_results_path.read_text())
    if not isinstance(results, list) or not results:
        return

    x = list(range(normalized_layer_count))
    plt.figure(figsize=(10, 5))

    for entry in results:
        saturation = entry.get("saturation") or {}
        if not saturation:
            continue

        parsed = []
        for key, value in saturation.items():
            match = re.search(r"(\d+)$", str(key))
            if not match:
                continue
            parsed.append((int(match.group(1)), float(value)))
        if not parsed:
            continue

        parsed.sort(key=lambda item: item[0])
        base = parsed[0][0]
        normalized_map = {}
        for raw_idx, sat_value in parsed:
            norm_idx = raw_idx - base
            if 0 <= norm_idx < normalized_layer_count:
                normalized_map[norm_idx] = sat_value

        y = [normalized_map.get(i, float("nan")) for i in x]
        label = (
            f"{voltage_alias}={entry.get('voltage_amp')}, "
            f"{current_alias}={entry.get('current_amp')}, "
            f"{iterations_alias}={entry.get('iterations')}"
        )
        plt.plot(x, y, marker="o", label=label)

    plt.xticks(x, [str(i) for i in x])
    plt.xlabel("Normalized layer index")
    plt.ylabel("Saturation (%)")
    plt.title(title)
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_max_gradients(x_axis, grad_data, output_dir):
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    for layer_name, tensor in grad_data.items():
        plt.figure(figsize=(8, 4))
        plt.plot(x_axis, tensor, label="max |grad|")
        plt.xlabel("Input value")
        plt.ylabel("Max |gradient|")
        plt.title(f"Max gradient for {layer_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"max_grad_{layer_name}.png")
        plt.close()


def plot_pca_sweep_grid(
    npz_path,
    output_dir=None,
    *,
    layers=None,
    reduce: str = "mean",
    feature_idx: Optional[int] = None,
    grid_size: Optional[int] = None,
    cmap: str = "viridis",
):
    """
    Plot PCA sweep states (saved by pca-sweep) as 2D grids.

    Each layer's state is reduced to a scalar per sample, then reshaped into a square grid.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    npz_path = Path(npz_path)
    data = np.load(npz_path)
    if layers is None:
        layer_names = list(data.files)
    elif isinstance(layers, (list, tuple, set)):
        layer_names = list(layers)
    else:
        layer_names = [layers]

    if output_dir is None:
        output_dir = npz_path.with_suffix("")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer_name in layer_names:
        if layer_name not in data:
            raise KeyError(f"Layer '{layer_name}' not found in {npz_path}")
        arr = data[layer_name]
        if arr.ndim < 1:
            raise ValueError(f"Layer '{layer_name}' has invalid shape {arr.shape}")
        flat = arr.reshape(arr.shape[0], -1)

        if feature_idx is not None:
            if feature_idx < 0 or feature_idx >= flat.shape[1]:
                raise IndexError(
                    f"feature_idx={feature_idx} out of range for layer '{layer_name}' (size={flat.shape[1]})"
                )
            values = flat[:, feature_idx]
            label = f"idx={feature_idx}"
        else:
            if reduce == "mean":
                values = flat.mean(axis=1)
            elif reduce == "max":
                values = flat.max(axis=1)
            elif reduce == "min":
                values = flat.min(axis=1)
            elif reduce == "norm":
                values = np.linalg.norm(flat, axis=1)
            elif reduce == "argmax":
                values = flat.argmax(axis=1)
            else:
                raise ValueError(f"Unsupported reduce='{reduce}'")
            label = reduce

        if grid_size is None:
            side = int(np.sqrt(values.shape[0]))
            if side * side != values.shape[0]:
                raise ValueError(
                    f"Sample count {values.shape[0]} is not a square; pass grid_size explicitly."
                )
        else:
            side = grid_size
            if side * side != values.shape[0]:
                raise ValueError(
                    f"grid_size={side} does not match sample count {values.shape[0]}."
                )

        grid = values.reshape(side, side)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(grid, origin="lower", cmap=cmap)
        fig.colorbar(im, ax=ax)
        ax.set_title(f"{layer_name} ({label})")
        ax.set_xlabel("PCA dim 1")
        ax.set_ylabel("PCA dim 2")
        safe_name = layer_name.replace("/", "_")
        fig.tight_layout()
        fig.savefig(output_dir / f"pca_grid_{safe_name}_{label}.png", dpi=150)
        plt.close(fig)


def flatten_history(history_list):
    tensor = torch.cat(history_list, dim=0)
    tensor = tensor.reshape(tensor.shape[0], -1)
    return tensor.numpy()


def flatten_vector_history(history_list):
    tensor = torch.cat(history_list, dim=0)
    return tensor.numpy()
