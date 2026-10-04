import argparse
from dataclasses import dataclass
import importlib.util
from itertools import product
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np
from sympy.core.kind import RaiseNotImplementedError
import torch
from torch.utils.data import DataLoader, TensorDataset

LABS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LABS_DIR.parent
DATASETS_MODULE_PATH = PROJECT_ROOT / "datasets.py"
DEFAULT_OUTPUT_ROOT = Path("/home/filip/server_code/simulation_results/experiments_labs")

for path in (LABS_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))
from training.lab.diagnostics import BetaSize
from training.core.batch import Batch
from training.core.engine import EvaluationComponents
from model.resistive.network import DeepResistiveEnergy  # noqa: E402
from model.function.network import Network  # noqa: E402
from labs.custom_minimizer import CustomQuadraticMinimizer, MinimizerSettings  # noqa: E402
from model.resistive.minimizer import QuadraticMinimizer  # noqa: E402
from training.core.sgd import AugmentedFunction, EquilibriumProp  # noqa: E402
from model.function.cost import SquaredError, SquaredErrorPairedOutputs  # noqa: E402
from labs.common import (
    MnistParts,
    save_beta_summary,
    build_evaluator,
    CustomEvaluator,
)
from supporting.stats_snapshot import snapshot_stats
from supporting.pca_utils import prepare_pca_grid
from labs.compute_pca import (
    compute_pca,
    _resolve_pca_cache_path,
    _load_pca_cache,
    _save_pca_cache,
)
from labs.mnist_io_runs import _init_run_dir
from labs.mnist_monitor_training import track_training_statistics

TRACK_FULL_HISTORY = False
DEFAULT_PCA_STEPS = 30
DEFAULT_PCA_N_SIGMA = 3.0
DEFAULT_PCA_BATCH_SIZE = None
DEFAULT_PCA_INCLUDE_IDX = False


@dataclass
class PcaGridSpec:
    steps: int
    n_sigma: float
    batch_size: Optional[int] = None
    include_idx: bool = False
    as_train_loader: bool = False
    cache_path: Optional[str] = None


def load_json_config(config_path):
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text())


def _truncate_batch(batch, n: int):
    if batch is None:
        return None
    if torch.is_tensor(batch):
        return batch[:n]
    if isinstance(batch, (list, tuple)):
        return type(batch)(_truncate_batch(x, n) for x in batch)
    return batch


def _subset_loader(loader: DataLoader, num_samples: Optional[int]):
    """Return a loader over the first num_samples items plus the actual sample count."""
    if num_samples is None:
        dataset_len = None
        if hasattr(loader, "dataset"):
            try:
                dataset_len = len(loader.dataset)
            except TypeError:
                dataset_len = None
        return loader, dataset_len

    if hasattr(loader, "dataset"):
        try:
            dataset_len = len(loader.dataset)
        except TypeError:
            dataset_len = None
        if dataset_len is not None and num_samples >= dataset_len:
            return loader, dataset_len
        if dataset_len is not None:
            subset = torch.utils.data.Subset(loader.dataset, list(range(num_samples)))
            subset_loader = DataLoader(
                subset,
                batch_size=getattr(loader, "batch_size", None),
                shuffle=False,
                num_workers=getattr(loader, "num_workers", 0),
                pin_memory=getattr(loader, "pin_memory", False),
                drop_last=False,
                collate_fn=getattr(loader, "collate_fn", None),
            )
            return subset_loader, num_samples

    # Fallback for iterable datasets: truncate at batch level.
    batches = []
    count = 0
    for batch in loader:
        if count >= num_samples:
            break
        if torch.is_tensor(batch):
            batch_size = batch.size(0)
        elif isinstance(batch, (list, tuple)) and batch:
            batch_size = batch[0].size(0) if torch.is_tensor(batch[0]) else len(batch[0])
        else:
            batch_size = 0
        take = min(num_samples - count, batch_size)
        batches.append(_truncate_batch(batch, take))
        count += take
        if count >= num_samples:
            break
    return batches, count


def _load_project_datasets_module():
    module_name = "ebl_datasets"
    if module_name in sys.modules:
        return sys.modules[module_name]
    if not DATASETS_MODULE_PATH.exists():
        raise FileNotFoundError(f"Expected datasets.py at {DATASETS_MODULE_PATH}")
    spec = importlib.util.spec_from_file_location(module_name, DATASETS_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load datasets module from {DATASETS_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module

load_dataloaders = _load_project_datasets_module().load_dataloaders


def prepare_mnist(
    config_path: str,
    model_key: str,
    voltage_amp: float = 1.0,
    current_amp: float = 1.0,
    weights_path: str = None,
    seed: int = None,
    *,
    data_mode: str = "mnist",
    train_loader_override=None,
    test_loader_override=None,
    pca: Optional[PcaGridSpec] = None,
    verbose: bool = False,
):
    """Common MNIST setup returning a bundled MnistParts."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_json_config(config_path)
    model_cfg = dict(config["models"][model_key])  # shallow copy for local overrides
    training_cfg = config.get("training", {})

    dataset_name = training_cfg["dataset"]
    batch_size = training_cfg["batch_size"]
    normalize = training_cfg["normalize"]
    normalize_std = training_cfg["normalize_std"]
    augment_32 = training_cfg["augment_32x32"]
    data_root = str(Path(training_cfg.get("data_root", "data")).expanduser())
    if verbose:
        print(
            f"[prepare_mnist] model={model_key} data_mode={data_mode} "
            f"dataset={dataset_name} batch_size={batch_size} voltage_amp={voltage_amp} current_amp={current_amp}"
        )

    default_train_loader = None
    default_test_loader = None
    if train_loader_override is None or test_loader_override is None:
        default_train_loader, default_test_loader = load_dataloaders(
            dataset_name,
            batch_size,
            augment_32x32=augment_32,
            normalize=normalize,
            normalize_std=normalize_std,
            data_root=data_root,
        )
    train_loader = default_train_loader if train_loader_override is None else train_loader_override
    test_loader = default_test_loader if test_loader_override is None else test_loader_override

    layer_shapes = [tuple(shape) for shape in model_cfg["layer_shapes"]]
    energy_fn = DeepResistiveEnergy(
        layer_shapes=layer_shapes,
        weight_gains=model_cfg["weight_gains"],
        input_gain=model_cfg.get("input_gain"),
        non_linearity=model_cfg["non_linearity"],
        exponential_diode_param=dict(model_cfg.get("exponential_diode_param", {})),
        quadratic_diode_param=dict(model_cfg.get("quadratic_diode_param", {})),
        hard_sigmoid_param=dict(model_cfg.get("hard_sigmoid_param", {})),
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        weight_min=model_cfg.get("weight_min"),
        weight_max=model_cfg.get("weight_max"),
        conv_pipeline=model_cfg.get("conv_pipeline"),
        pooling_mode=model_cfg.get("pooling_mode"),
    )
    energy_fn.set_device(device)
    if weights_path:
        energy_fn.load(weights_path)

    network = Network(energy_fn)
    free_layers = network.free_layers()
    output_layer = energy_fn.layers()[-1]
    if output_layer.shape[0] == 10:
        cost_fn = SquaredError(output_layer)
    elif output_layer.shape[0] == 20:
        cost_fn = SquaredErrorPairedOutputs(output_layer, 10)
    else:
        raise ValueError(f"Unsupported output size: {output_layer.shape}")

    minimizer_mode = config["energy_minimizer"]["mode"]

    # Learning rates are model-specific in this codebase and live under the model config.
    learning_rates = None
    if "learning_rates" in model_cfg:
        learning_rates = list(model_cfg["learning_rates"])
    elif "learning_rates_biases" in model_cfg and "learning_rates_weights" in model_cfg:
        learning_rates = list(model_cfg["learning_rates_biases"]) + list(model_cfg["learning_rates_weights"])

    if test_loader_override is None and data_mode == "pca_grid":
        if pca is None:
            raise ValueError("pca spec is required when data_mode='pca_grid'.")
        cache_path = _resolve_pca_cache_path(pca.cache_path, model_key, dataset_name)
        if cache_path and cache_path.exists():
            mu, eigvals_2, eigvecs_2, n_total = _load_pca_cache(cache_path)
            if verbose:
                print(f"[prepare_mnist] loaded PCA cache from {cache_path}")
        else:
            mu, eigvals_2, eigvecs_2, n_total = compute_pca(config_path, model_key)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                _save_pca_cache(cache_path, mu, eigvals_2, eigvecs_2, n_total)
                if verbose:
                    print(f"[prepare_mnist] saved PCA cache to {cache_path}")
        side = 32 if augment_32 else 28
        pca_loader_batch_size = batch_size if pca.batch_size is None else pca.batch_size
        pca_loader = prepare_pca_grid(
            eigvals_2,
            eigvecs_2,
            mu,
            n_total,
            steps=pca.steps,
            n_sigma=pca.n_sigma,
            image_shape=(1, side, side),
            batch_size=pca_loader_batch_size,
        )
        if verbose:
            dataset_len = len(pca_loader.dataset) if hasattr(pca_loader, "dataset") else "?"
            print(
                f"[prepare_mnist] PCA grid prepared: steps={pca.steps} n_sigma={pca.n_sigma} "
                f"points={dataset_len} batch_size={pca_loader_batch_size} n_total={n_total}"
            )
        if pca.include_idx:
            x_tensor, y_tensor = pca_loader.dataset.tensors[:2]
            idx_tensor = torch.arange(x_tensor.shape[0], dtype=torch.long)
            pca_loader = DataLoader(
                TensorDataset(x_tensor, y_tensor, idx_tensor),
                batch_size=pca_loader.batch_size,
                shuffle=False,
            )
        test_loader = pca_loader
        if pca.as_train_loader:
            train_loader = pca_loader

    return MnistParts(
        energy_fn=energy_fn,
        network=network,
        cost_fn=cost_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        free_layers=free_layers,
        minimizer_mode=minimizer_mode,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        learning_rates=learning_rates,
        scheduler=None,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        weights_path=weights_path,
        energy_minimizer_cfg=dict(config["energy_minimizer"]),
    )


def _resolve_double_diode_runtime(double_diode_runtime: str) -> tuple[str, bool]:
    if double_diode_runtime == "float64_fixed":
        return "float64_timed", False
    if double_diode_runtime == "float64_experimental":
        return "float64_experimental", False
    if double_diode_runtime == "float32_adaptive":
        return "float32", True
    if double_diode_runtime == "float32_overrelaxed":
        return "overrelaxed", True
    raise ValueError(
        "double_diode_runtime must be one of: float32_adaptive, float64_fixed, "
        "float64_experimental, float32_overrelaxed."
    )


def _build_minimizer(
    parts: MnistParts,
    num_iterations: int,
    *,
    fn=None,
    overrides: Optional[dict] = None,
    double_diode_runtime: Optional[str] = None,
):
    energy_fn = parts.energy_fn if fn is None else fn
    if parts.energy_minimizer_cfg is not None:
        minimizer_cfg = dict(parts.energy_minimizer_cfg)
    else:
        minimizer_cfg = dict(parts.training_cfg.get("energy_minimizer", {}))
    if double_diode_runtime:
        dd_updater, adaptive_equilibrium = _resolve_double_diode_runtime(double_diode_runtime)
        minimizer_cfg["double_diode_updater"] = dd_updater
        minimizer_cfg["adaptive_equilibrium"] = adaptive_equilibrium
    if overrides:
        minimizer_cfg.update(overrides)
    adaptive_equilibrium = minimizer_cfg.get("adaptive_equilibrium", True)
    rel_tol = minimizer_cfg.get("rel_tol", 1e-5)
    vn_tol = minimizer_cfg.get("vn_tol", 1e-6)
    use_polish = minimizer_cfg.get("use_polish", True)
    max_newton_iters = minimizer_cfg.get("max_newton_iters", 32)
    z_thresh = minimizer_cfg.get("z_thresh", 1e10)
    dynamic_polish = minimizer_cfg.get("dynamic_polish", True)
    overrelaxation_reject_steps = minimizer_cfg.get("overrelaxation_reject_steps", False)
    overrelaxation_reject_max_tries = minimizer_cfg.get("overrelaxation_reject_max_tries", 3)
    overrelaxation_reject_shrink = minimizer_cfg.get("overrelaxation_reject_shrink", 0.5)
    overrelaxation_reject_eps = minimizer_cfg.get("overrelaxation_reject_eps", 0.0)
    double_diode_updater = minimizer_cfg.get("double_diode_updater", "CustomExponentialDoubleDiodeUpdater")
    overrelaxation_factor = minimizer_cfg.get("overrelaxation_factor", 1.1)
    single_diode_updater = minimizer_cfg.get("single_diode_updater", "custom")
    iv_data = minimizer_cfg.get("iv_data")
    iv_data_path = minimizer_cfg.get("iv_data_path")
    exp_clip_default = 100000.0
    if parts.model_cfg.get("non_linearity") == "double_diode_exponential":
        if double_diode_updater in ("float32", "overrelaxed"):
            exp_clip_default = 80.0
        elif double_diode_updater in ("float64_timed", "TimedExponentialDOubleDiodeUpdater"):
            exp_clip_default = 10000.0
        else:
            exp_clip_default = 100000.0
    exp_clip = minimizer_cfg.get("exp_clip", exp_clip_default)
    minimizer_settings = MinimizerSettings(
        rel_tol=rel_tol,
        vn_tol=vn_tol,
        use_polish=use_polish,
        max_newton_iters=max_newton_iters,
        z_thresh=z_thresh,
        exp_clip=exp_clip,
        dynamic_polish=dynamic_polish,
        overrelaxation_reject_steps=overrelaxation_reject_steps,
        overrelaxation_reject_max_tries=overrelaxation_reject_max_tries,
        overrelaxation_reject_shrink=overrelaxation_reject_shrink,
        overrelaxation_reject_eps=overrelaxation_reject_eps,
    )
    return CustomQuadraticMinimizer(
        fn=energy_fn,
        free_layers=parts.free_layers,
        num_iterations=num_iterations,
        mode=parts.minimizer_mode,
        non_linearity=parts.model_cfg["non_linearity"],
        quadratic_diode_param=parts.model_cfg.get("quadratic_diode_param", {}),
        exponential_diode_param=parts.model_cfg.get("exponential_diode_param", {}),
        hard_sigmoid_param=parts.model_cfg.get("hard_sigmoid_param", {}),
        voltage_amp=parts.energy_fn._voltage_amp,
        current_amp=parts.energy_fn._current_amp,
        iv_data=iv_data,
        iv_data_path=iv_data_path,
        double_diode_updater=double_diode_updater,
        adaptive_equilibrium=adaptive_equilibrium,
        overrelaxation_factor=overrelaxation_factor,
        single_diode_updater=single_diode_updater,
        minimizer_settings=minimizer_settings,
    )


def _build_classic_minimizer(parts: MnistParts, num_iterations: int, *, fn=None):
    energy_fn = parts.energy_fn if fn is None else fn
    return QuadraticMinimizer(
        fn=energy_fn,
        free_layers=parts.free_layers,
        num_iterations=num_iterations,
        mode=parts.minimizer_mode,
        non_linearity=parts.model_cfg["non_linearity"],
        quadratic_diode_param=parts.model_cfg.get("quadratic_diode_param", {}),
        exponential_diode_param=parts.model_cfg.get("exponential_diode_param", {}),
        hard_sigmoid_param=parts.model_cfg.get("hard_sigmoid_param", {}),
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
    )

def validate_mnist(
    parts: MnistParts,
    num_iterations: int,
    *,
    num_samples: int = 1000,
    weights_path: Optional[str] = None,
    output_dir: Optional[Path] = None,
    output_name: str = "validation_layer_states.npz",
    verbose: bool = False,
    show_running_accuracy: bool = True,
    build_minimizer_fn=None,
):
    """
    Evaluate the first N samples from the MNIST test loader and save layer states.
    """
    energy_fn = parts.energy_fn
    network = parts.network
    cost_fn = parts.cost_fn
    load_path = weights_path or parts.weights_path
    if load_path:
        energy_fn.load(load_path)

    test_loader = parts.test_loader
    if isinstance(test_loader, DataLoader) and getattr(test_loader, "num_workers", 0):
        test_loader = DataLoader(
            test_loader.dataset,
            batch_size=getattr(test_loader, "batch_size", None),
            shuffle=False,
            num_workers=0,
            pin_memory=getattr(test_loader, "pin_memory", False),
            drop_last=False,
            collate_fn=getattr(test_loader, "collate_fn", None),
        )
    test_loader, actual_count = _subset_loader(test_loader, num_samples)
    if build_minimizer_fn is None:
        build_minimizer_fn = _build_minimizer
    minimizer = build_minimizer_fn(parts, int(num_iterations))

    layer_states = {}
    correct = 0
    total = 0
    for batch in test_loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0] if len(batch) > 0 else None
            y = batch[1] if len(batch) > 1 else None
        else:
            x = batch
            y = None
        if x is None:
            continue

        network.set_input(x, reset=True)
        minimizer.compute_equilibrium()

        if y is not None:
            cost_fn.set_target(y)
            errors = cost_fn.error_fn()
            correct += int((~errors).sum().item())
            total += int(errors.numel())
            if show_running_accuracy and total > 0:
                acc = correct / total
                print(
                    f"\r[validate_mnist] {total}/{actual_count} accuracy={acc:.4f}",
                    end="",
                    flush=True,
                )

        for i, layer in enumerate(network.layers()):
            name = getattr(layer, "name", f"layer_{i}")
            layer_states.setdefault(name, []).append(layer.state.detach().cpu().clone())

        if verbose:
            print(f"\r[validate_mnist] processed {total}/{actual_count}", end="", flush=True)

    if show_running_accuracy and total > 0:
        print()
    elif verbose:
        print()

    arrays = {}
    for name, values in layer_states.items():
        if values:
            arrays[name] = torch.cat(values, dim=0).detach().cpu().numpy()

    output_root = Path(output_dir) if output_dir is not None else Path.cwd()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / output_name
    np.savez(output_path, **arrays)

    return {
        "num_samples": actual_count,
        "output_path": str(output_path),
        "layer_states": list(arrays.keys()),
        "accuracy": (correct / total) if total > 0 else None,
    }


def validate_mnist_float64_fixed(
    parts: MnistParts,
    num_iterations: int,
    *,
    num_samples: int = 1000,
    weights_path: Optional[str] = None,
    output_dir: Optional[Path] = None,
    output_name: str = "validation_layer_states_float64_fixed.npz",
    verbose: bool = False,
    show_running_accuracy: bool = True,
):
    """Validate MNIST with float64_fixed runtime and adaptive equilibrium disabled."""
    def _build_fixed(parts_in: MnistParts, num_iter: int):
        return _build_minimizer(
            parts_in,
            num_iter,
            double_diode_runtime="float64_fixed",
            overrides={"adaptive_equilibrium": False},
        )

    return validate_mnist(
        parts,
        num_iterations,
        num_samples=num_samples,
        weights_path=weights_path,
        output_dir=output_dir,
        output_name=output_name,
        verbose=verbose,
        show_running_accuracy=show_running_accuracy,
        build_minimizer_fn=_build_fixed,
    )


def validate_mnist_classic(
    parts: MnistParts,
    num_iterations: int,
    *,
    num_samples: int = 1000,
    weights_path: Optional[str] = None,
    output_dir: Optional[Path] = None,
    output_name: str = "validation_layer_states_classic.npz",
    verbose: bool = False,
    show_running_accuracy: bool = True,
):
    """Validate MNIST with the classic QuadraticMinimizer."""
    return validate_mnist(
        parts,
        num_iterations,
        num_samples=num_samples,
        weights_path=weights_path,
        output_dir=output_dir,
        output_name=output_name,
        verbose=verbose,
        show_running_accuracy=show_running_accuracy,
        build_minimizer_fn=_build_classic_minimizer,
    )


def validate_spice_results(spice_npz_file, test_dataloader):
    spice_npz = Path(spice_npz_file)
    with np.load(spice_npz) as data:

        if 'Layer_2' in data.files:
            results = data['Layer_2']
        else:
            raise ValueError("Layer_2 does not exist")
    preds = []
    for result in results:
        even_values = result[0::2]
        odd_values = result[1::2]
        prediction = np.argmax(even_values-odd_values)
        preds.append(prediction)

    preds = np.array(preds)
    labels = []
    for batch in test_dataloader:
        if len(batch) == 3:
            _, y, _ = batch
        else:
            _, y = batch

        labels.append(y.numpy())
    labels = np.concatenate(labels)

    accuracy = (preds == labels[:len(preds)]).mean()
    print(accuracy)


def _compute_beta_summary(
    parts: MnistParts,
    num_iterations: int,
    *,
    max_batches: Optional[int] = None,
    verbose: bool = False,
    nudging_override: Optional[float] = None,
) -> dict:
    """Measure free->nudged displacement (beta-size) on parts.test_loader."""
    energy_fn = parts.energy_fn
    network = parts.network
    cost_fn = parts.cost_fn
    model_cfg = parts.model_cfg
    training_cfg = parts.training_cfg
    params = energy_fn.params()

    energy_minimizer_inference = _build_minimizer(parts, num_iterations)
    augmented_fn = AugmentedFunction(energy_fn, cost_fn)
    energy_minimizer_training = _build_minimizer(parts, num_iterations, fn=augmented_fn)

    nudging = nudging_override if nudging_override is not None else model_cfg.get("nudging", training_cfg.get("nudging"))
    ep_variant = training_cfg.get("ep_variant")
    ep_kwargs = {}
    if nudging is not None:
        ep_kwargs["nudging"] = nudging
    if ep_variant is not None:
        ep_kwargs["variant"] = ep_variant
    estimator = EquilibriumProp(
        params, parts.free_layers, augmented_fn, cost_fn, energy_minimizer_training, **ep_kwargs
    )

    beta = BetaSize(
        EvaluationComponents(network, cost_fn, energy_minimizer_inference),
        parts.test_loader,
        estimator,
        max_batches=max_batches,
    )
    beta.run(verbose=verbose)
    beta_summary = beta.summary()
    layer_names = beta_summary.get("layers", [])
    if layer_names:
        beta_summary["layers"] = [f"Layer_{i}" for i, _ in enumerate(layer_names)]
    return beta_summary


def prepare_mnist_parts_sweep(
    config_path: str,
    model_key: str,
    voltage_amps=(1.0,),
    current_amps=(1.0,),
    iteration_counts=(12,),
    weights_path: str = None,
    seed: int = None,
    *,
    data_mode: str = "pca_grid",
    pca: Optional[PcaGridSpec] = None,
    verbose: bool = True,
):
    """Build one MnistParts per sweep point (amps x iteration_counts)."""
    voltage_values = tuple(voltage_amps)
    current_values = tuple(current_amps)
    iteration_values = tuple(iteration_counts)
    if not voltage_values or not current_values or not iteration_values:
        raise ValueError("voltage_amps, current_amps and iteration_counts must be non-empty.")

    total_points = len(voltage_values) * len(current_values) * len(iteration_values)
    if verbose:
        print(
            f"[prepare_mnist_parts_sweep] building {total_points} sweep points "
            f"(voltage={voltage_values}, current={current_values}, iterations={iteration_values})"
        )

    # Build loaders once and reuse them across sweep points so the PCA grid is not recomputed.
    shared_train_loader = None
    shared_test_loader = None
    if data_mode == "pca_grid":
        if pca is None:
            raise ValueError("pca spec is required when data_mode='pca_grid'.")
        effective_pca = PcaGridSpec(
            steps=pca.steps,
            n_sigma=pca.n_sigma,
            batch_size=pca.batch_size,
            include_idx=True,
            as_train_loader=pca.as_train_loader,
            cache_path=pca.cache_path,
        )
        if verbose and not pca.include_idx:
            print("[prepare_mnist_parts_sweep] forcing pca.include_idx=True for evaluator compatibility.")
        shared_parts = prepare_mnist(
            config_path=config_path,
            model_key=model_key,
            voltage_amp=float(voltage_values[0]),
            current_amp=float(current_values[0]),
            weights_path=None,
            seed=seed,
            data_mode=data_mode,
            pca=effective_pca,
            verbose=False,
        )
        shared_train_loader = shared_parts.train_loader
        shared_test_loader = shared_parts.test_loader

    parts_list = []
    for idx, (voltage_amp, current_amp, num_iterations) in enumerate(
        product(voltage_values, current_values, iteration_values),
        start=1,
    ):
        if verbose:
            print(
                f"[prepare_mnist_parts_sweep] [{idx}/{total_points}] "
                f"v={float(voltage_amp):g} c={float(current_amp):g} iters={int(num_iterations)}"
            )
        parts = prepare_mnist(
            config_path=config_path,
            model_key=model_key,
            voltage_amp=float(voltage_amp),
            current_amp=float(current_amp),
            weights_path=weights_path,
            seed=seed,
            data_mode=data_mode,
            train_loader_override=shared_train_loader,
            test_loader_override=shared_test_loader,
            pca=effective_pca if data_mode == "pca_grid" else pca,
            verbose=False,
        )
        parts.num_iterations = int(num_iterations)
        parts_list.append(parts)
    return parts_list


def sweep_MnistParts(
    sweep_parts,
    weights_path=None,
    record_statistics=("calc_residual_current",),
    verbose=True,
    output_dir: Optional[Path] = None,
    track_beta: bool = False,
    beta_max_batches: Optional[int] = None,
    beta_only: bool = False,
    beta_nudging: Optional[float] = None,
    record: Optional[str] = None,
):
    """Run a single PCA sweep per MnistParts and collect requested outputs."""
    sweep_parts = list(sweep_parts)
    results = []
    total_points = len(sweep_parts)
    if beta_only and not track_beta:
        raise ValueError("beta_only=True requires track_beta=True.")
    if record is None:
        raise ValueError("record is required; choose one of: stats, residuals, states.")
    if record not in ("stats", "residuals", "states"):
        raise ValueError("record must be one of: stats, residuals, states.")
    for idx, parts in enumerate(sweep_parts, start=1):
        energy_fn = parts.energy_fn
        load_path = weights_path or parts.weights_path
        if load_path:
            energy_fn.load(load_path)

        if parts.num_iterations is None:
            raise ValueError("Each MnistParts in sweep_parts must define num_iterations.")
        if verbose:
            pca_batch = getattr(parts.test_loader, "batch_size", None)
            print(
                f"[sweep_MnistParts] [{idx}/{total_points}] "
                f"v={parts.voltage_amp:g} c={parts.current_amp:g} iters={parts.num_iterations} "
                f"test_batch_size={pca_batch}"
            )

        stats_snapshot = {}
        residual_current = {}
        layer_states = {}
        if not beta_only:
            sweep_result = sweep_pca(
                parts,
                parts.num_iterations,
                output=record,
                verbose=verbose,
                record_statistics=record_statistics,
            )
            if record == "stats":
                stats_snapshot = sweep_result or {}
            elif record == "residuals":
                residual_current = sweep_result or {}
            elif record == "states":
                layer_states = sweep_result or {}
        result = {
            "voltage_amp": parts.voltage_amp,
            "current_amp": parts.current_amp,
            "iterations": parts.num_iterations,
        }
        if stats_snapshot:
            result["stats_snapshot"] = stats_snapshot
        if residual_current:
            result["residual_current"] = residual_current
        if track_beta:
            result["beta_summary"] = _compute_beta_summary(
                parts,
                num_iterations=int(parts.num_iterations),
                max_batches=beta_max_batches,
                verbose=verbose,
                nudging_override=beta_nudging,
            )
        if record == "states":
            result["layer_states"] = layer_states

        if output_dir is not None:
            point_dir = output_dir / (
                f"v{parts.voltage_amp:g}_c{parts.current_amp:g}_iter{parts.num_iterations}"
            )
            point_dir.mkdir(parents=True, exist_ok=True)
            if stats_snapshot:
                (point_dir / "stats_snapshot.json").write_text(json.dumps(stats_snapshot, indent=2))
                result["stats_path"] = str(point_dir / "stats_snapshot.json")
            if residual_current:
                (point_dir / "residual_current.json").write_text(json.dumps(residual_current, indent=2))
                result["residual_current_path"] = str(point_dir / "residual_current.json")
            if track_beta and "beta_summary" in result:
                save_beta_summary(result["beta_summary"], point_dir)
                result["beta_summary_path"] = str(point_dir / "beta_summary.json")
            if verbose:
                print(f"[sweep_MnistParts] wrote outputs to {point_dir}")

            if record == "states" and layer_states:
                arrays = {}
                for name, values in layer_states.items():
                    if values:
                        arrays[name] = torch.cat(values, dim=0).detach().cpu().numpy()
                if arrays:
                    layer_states_path = point_dir / "layer_states.npz"
                    np.savez(layer_states_path, **arrays)
                    result["layer_states_path"] = str(layer_states_path)

        results.append(result)

    if verbose:
        print(f"[sweep_MnistParts] finished {len(results)} sweep points")

    return results


def sweep_pca(
    parts: MnistParts,
    num_of_iterations,
    *,
    output: str = "states",
    verbose: bool = False,
    record_statistics=("calc_residual_current",),
):
    """
    Run inference over a PCA grid loader and collect either states or residual currents.

    Args:
        output: "states" (per-layer settled states) or "residuals" (per-layer residual currents).
    """
    energy_minimizer = _build_minimizer(parts, int(num_of_iterations))

    if output == "stats":
        evaluator = build_evaluator(
            parts.network,
            parts.cost_fn,
            parts.test_loader,
            energy_minimizer,
            parts.model_cfg,
            record_statistics,
        )
        evaluator.run(verbose=verbose)
        return snapshot_stats([evaluator.statistics])

    if output in ("states", "residuals"):
        def input_batches():
            # PCA loaders may yield tensors, input-only tuples, or labelled
            # batches. State/residual diagnostics do not require targets.
            for raw in parts.test_loader:
                if isinstance(raw, Batch):
                    inputs = raw.inputs
                elif isinstance(raw, (tuple, list)):
                    inputs = raw[0] if raw else None
                else:
                    inputs = raw
                if inputs is not None:
                    yield Batch(inputs, None)

        requested = "store_states" if output == "states" else "calc_residual_current"
        evaluator = CustomEvaluator(
            EvaluationComponents(parts.network, None, energy_minimizer),
            input_batches(), reset_input=True, record_statistics=(requested,),
        )
        evaluator.run(verbose=verbose)
        if output == "states":
            return {
                name: torch.cat(values, dim=0)
                for name, values in evaluator.layer_states.items()
                if values
            }
        return evaluator.res_currents

    raise ValueError("output must be 'stats', 'states' or 'residuals'.")


def _build_arg_parser() -> argparse.ArgumentParser:
    """CLI for running small MNIST experiments from this module."""
    p = argparse.ArgumentParser(prog="mnist_tests.py")

    # Common experiment wiring (kept shared across subcommands).
    p.add_argument("--config", required=True, help="Path to JSON config (e.g. config.used.json).")
    p.add_argument("--model-key", required=True, help="Key under config['models'] (e.g. drn-conv).")
    p.add_argument("--weights", default=None, help="Optional path to saved model weights (.pt).")
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for run outputs.",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    p.add_argument("--num-iterations", type=int, required=True, help="Minimizer iterations.")
    p.add_argument("--voltage-amp", type=float, default=1.0, help="Voltage amplification factor.")
    p.add_argument("--current-amp", type=float, default=1.0, help="Current amplification factor.")
    p.add_argument("--verbose", action="store_true", help="Verbose per-batch logging.")
    p.add_argument(
        "--pca-cache",
        default=None,
        help=(
            "Optional .pt path to cache PCA components. "
            "If omitted and using PCA grid, defaults to labs/datasets/pca_<dataset>_<model>.pt."
        ),
    )

    sub = p.add_subparsers(dest="cmd")
    p.set_defaults(cmd="train-stats")

    train_stats = sub.add_parser("train-stats", help="Train for a few epochs while tracking residual currents.")
    train_stats.add_argument("--num-epochs", type=int, default=None, help="Number of training epochs (default: 1 if missing).")
    train_stats.add_argument(
        "--record-statistics",
        nargs="*",
        default=("calc_residual_current",),
        help="Optional recordings (e.g. calc_residual_current).",
    )
    train_stats.add_argument(
        "--out",
        default=None,
        help="Optional JSON path to write residual currents (if omitted, prints a short summary).",
    )

    validate_fixed = sub.add_parser(
        "validate-fixed64",
        help="Run MNIST validation with float64_fixed runtime (adaptive equilibrium disabled).",
    )
    validate_fixed.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of MNIST test samples to evaluate.",
    )
    validate_fixed.add_argument(
        "--output-name",
        default="validation_layer_states_float64_fixed.npz",
        help="Filename for saved layer states.",
    )

    validate_classic = sub.add_parser(
        "validate-classic",
        help="Run MNIST validation with the classic QuadraticMinimizer.",
    )
    validate_classic.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of MNIST test samples to evaluate.",
    )
    validate_classic.add_argument(
        "--output-name",
        default="validation_layer_states_classic.npz",
        help="Filename for saved layer states.",
    )

    sub.add_parser(
        "pca-sweep",
        help="Run inference over a PCA grid and save layer states/residual currents.",
    )

    pca_sweep_loop = sub.add_parser(
        "pca-sweep-loop",
        help="Run PCA sweeps over one selected variable and save each value to a folder.",
    )
    pca_sweep_loop.add_argument(
        "--swept_variable",
        choices=("voltage_amp", "current_amp", "num_iterations"),
        required=True,
        help="Variable to sweep.",
    )
    pca_sweep_loop.add_argument(
        "--sweep_values",
        nargs="+",
        required=True,
        help="Values for the swept variable (space-separated).",
    )

    return p


def _parse_sweep_values(raw_values, swept_variable):
    if swept_variable == "num_iterations":
        values = tuple(int(v) for v in raw_values)
        if any(v <= 0 for v in values):
            raise ValueError("num_iterations sweep values must be > 0.")
        return values
    return tuple(float(v) for v in raw_values)


def _format_swept_value(value) -> str:
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).replace("/", "_")


def main(argv: Optional[list[str]] = None) -> int:
    """
    Thin CLI dispatcher.

    Keep experiment logic in helpers (`prepare_mnist`, `track_training_statistics`, ...),
    and keep this function focused on: parse args → build parts → dispatch → save minimal outputs.
    """
    args = _build_arg_parser().parse_args(argv)
    run_dir = _init_run_dir(args)

    if args.cmd == "train-stats":
        parts = prepare_mnist(
            config_path=args.config,
            model_key=args.model_key,
            voltage_amp=args.voltage_amp,
            current_amp=args.current_amp,
            weights_path=args.weights,
            seed=args.seed,
            verbose=args.verbose,
        )
        res_currents = track_training_statistics(
            parts,
            num_iterations=args.num_iterations,
            weights_path=args.weights,
            build_minimizer_fn=_build_minimizer,
            record_statistics=tuple(args.record_statistics),
            verbose=args.verbose,
            num_epochs=args.num_epochs,
            output_dir=run_dir,
        )
        out_path = Path(args.out).expanduser() if args.out else run_dir / "residual_currents.json"
        if not out_path.is_absolute():
            out_path = run_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res_currents, indent=2))
        train_layers = len(res_currents.get("train", {}))
        test_layers = len(res_currents.get("test", {}))
        print(f"Wrote residual currents to {out_path} (train layers={train_layers}, test layers={test_layers}).")
        return 0

    if args.cmd == "validate-fixed64":
        parts = prepare_mnist(
            config_path=args.config,
            model_key=args.model_key,
            voltage_amp=args.voltage_amp,
            current_amp=args.current_amp,
            weights_path=args.weights,
            seed=args.seed,
            verbose=args.verbose,
        )
        result = validate_mnist_float64_fixed(
            parts,
            num_iterations=args.num_iterations,
            num_samples=args.num_samples,
            weights_path=args.weights,
            output_dir=run_dir,
            output_name=args.output_name,
            verbose=args.verbose,
            show_running_accuracy=True,
        )
        print(f"[validate-fixed64] wrote {result['output_path']}")
        if result.get("accuracy") is not None:
            print(f"[validate-fixed64] accuracy={result['accuracy']:.4f}")
        return 0

    if args.cmd == "validate-classic":
        parts = prepare_mnist(
            config_path=args.config,
            model_key=args.model_key,
            voltage_amp=args.voltage_amp,
            current_amp=args.current_amp,
            weights_path=args.weights,
            seed=args.seed,
            verbose=args.verbose,
        )
        result = validate_mnist_classic(
            parts,
            num_iterations=args.num_iterations,
            num_samples=args.num_samples,
            weights_path=args.weights,
            output_dir=run_dir,
            output_name=args.output_name,
            verbose=args.verbose,
            show_running_accuracy=True,
        )
        print(f"[validate-classic] wrote {result['output_path']}")
        if result.get("accuracy") is not None:
            print(f"[validate-classic] accuracy={result['accuracy']:.4f}")
        return 0

    if args.cmd == "pca-sweep":
        parts = prepare_mnist(
            config_path=args.config,
            model_key=args.model_key,
            voltage_amp=args.voltage_amp,
            current_amp=args.current_amp,
            weights_path=args.weights,
            seed=args.seed,
            data_mode="pca_grid",
            pca=PcaGridSpec(
                steps=DEFAULT_PCA_STEPS,
                n_sigma=DEFAULT_PCA_N_SIGMA,
                batch_size=DEFAULT_PCA_BATCH_SIZE,
                include_idx=DEFAULT_PCA_INCLUDE_IDX,
                cache_path=args.pca_cache,
            ),
            verbose=args.verbose,
        )
        pca_samples = None
        if hasattr(parts.test_loader, "dataset"):
            try:
                pca_samples = len(parts.test_loader.dataset)
            except TypeError:
                pca_samples = None
        start_time = time.perf_counter()
        layer_states = sweep_pca(parts, args.num_iterations, output="states", verbose=args.verbose)
        elapsed = time.perf_counter() - start_time
        states_path = run_dir / f"pca_sweep_iter{args.num_iterations}.npz"
        states_path.parent.mkdir(parents=True, exist_ok=True)
        state_arrays = {name: tensor.detach().cpu().numpy() for name, tensor in layer_states.items()}
        np.savez(states_path, **state_arrays)
        print(f"Wrote PCA sweep states to {states_path}")

        residuals = sweep_pca(parts, args.num_iterations, output="residuals", verbose=args.verbose)
        residuals_path = run_dir / f"residuals_iter{args.num_iterations}.npz"
        np.savez(residuals_path, **{k: np.asarray(v) for k, v in residuals.items()})
        print(f"Wrote PCA sweep residuals to {residuals_path}")
        if pca_samples:
            per_sample = elapsed / pca_samples
            print(
                f"[pca-sweep] elapsed={elapsed:.3f}s samples={pca_samples} "
                f"avg_per_sample={per_sample:.6f}s"
            )
            try:
                dataset_name = parts.training_cfg.get("dataset", "MNIST")
                info_path = run_dir / "run_info.json"
                info = json.loads(info_path.read_text()) if info_path.exists() else {}
                info.update(
                    {
                        "pca_sweep_seconds": elapsed,
                        "pca_samples": pca_samples,
                        "pca_seconds_per_sample": per_sample,
                        "pca_steps": DEFAULT_PCA_STEPS,
                        "pca_n_sigma": DEFAULT_PCA_N_SIGMA,
                        "pca_batch_size": DEFAULT_PCA_BATCH_SIZE,
                        "pca_num_iterations": args.num_iterations,
                        "pca_cache_path": str(
                            _resolve_pca_cache_path(args.pca_cache, args.model_key, dataset_name)
                        ),
                    }
                )
                info_path.write_text(json.dumps(info, indent=2))
            except Exception as exc:
                print(f"[pca-sweep] warning: failed to update run_info.json: {exc}")
        return 0

    if args.cmd == "pca-sweep-loop":
        sweep_values = _parse_sweep_values(args.sweep_values, args.swept_variable)
        voltage_amps = (args.voltage_amp,)
        current_amps = (args.current_amp,)
        iteration_counts = (args.num_iterations,)
        if args.swept_variable == "voltage_amp":
            voltage_amps = sweep_values
        elif args.swept_variable == "current_amp":
            current_amps = sweep_values
        else:
            iteration_counts = sweep_values

        sweep_parts = prepare_mnist_parts_sweep(
            config_path=args.config,
            model_key=args.model_key,
            voltage_amps=voltage_amps,
            current_amps=current_amps,
            iteration_counts=iteration_counts,
            weights_path=args.weights,
            seed=args.seed,
            data_mode="pca_grid",
            pca=PcaGridSpec(
                steps=DEFAULT_PCA_STEPS,
                n_sigma=DEFAULT_PCA_N_SIGMA,
                batch_size=DEFAULT_PCA_BATCH_SIZE,
                include_idx=DEFAULT_PCA_INCLUDE_IDX,
                cache_path=args.pca_cache,
            ),
            verbose=args.verbose,
        )

        base_dir = run_dir / "pca_sweep_loop"
        base_dir.mkdir(parents=True, exist_ok=True)
        for parts in sweep_parts:
            if not hasattr(parts, args.swept_variable):
                raise AttributeError(
                    f"MnistParts has no attribute '{args.swept_variable}' for folder naming."
                )
            swept_value = getattr(parts, args.swept_variable)
            value_tag = _format_swept_value(swept_value)

            layer_states = sweep_pca(parts, parts.num_iterations, output="states", verbose=args.verbose)
            iter_dir = base_dir / f"{args.swept_variable}_{value_tag}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            out_path = iter_dir / f"pca_sweep_iter{parts.num_iterations}.npz"
            arrays = {name: tensor.detach().cpu().numpy() for name, tensor in layer_states.items()}
            np.savez(out_path, **arrays)
            print(f"Wrote PCA sweep states to {out_path}")
            residuals = sweep_pca(parts, parts.num_iterations, output="residuals", verbose=args.verbose)
            residuals_path = iter_dir / f"residuals_iter{parts.num_iterations}.npz"
            np.savez(residuals_path, **{k: np.asarray(v) for k, v in residuals.items()})
            print(f"Wrote PCA sweep residuals to {residuals_path}")
        return 0

    raise RuntimeError(f"Unhandled command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
