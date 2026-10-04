import argparse
from datetime import datetime
import hashlib
import json
import math
import os
import random
import socket
import sys
from importlib import import_module
import importlib.util
from numbers import Integral
from pathlib import Path

import torch
import numpy as np

try:  # Optional TensorBoard support
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


# Ensure project modules are importable
LABS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LABS_DIR.parent / "energy-based-learning"
DEFAULT_IMAGE_RUNS_BASE = LABS_DIR.parent / "simulation_results" / "experiments_labs"

for path in (LABS_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))


from custom_classes import (
    ConvLayer,
    ConvResistive,
    DetailedSumSeparableFunction,
    FlexibleDeepResistiveEnergy,
    TrackingQuadraticMinimizer,
)  # noqa: E402
from custom_minimizer import CustomQuadraticMinimizer as QuadraticMinimizer, MinimizerSettings  # noqa: E402
from model.function.cost import SquaredError, SquaredErrorPairedOutputs  # noqa: E402
from model.function.network import Network  # noqa: E402
from model.variable.parameter import ConvWeight  # noqa: E402
from training.sgd import AugmentedFunction, EquilibriumProp  # noqa: E402
from training.optimizers import build_optimizer  # noqa: E402


def _default_quadratic_params():
    return {"diode_conductance": 1.0, "v_min": -1e6, "v_max": 1e6}


def _default_exponential_params():
    return {"I_s": 1e-6, "V_t": 0.025, "V_off": 0.0}


def _validate_optional_batch_limit(name, value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(
            f"Expected {name} to be null or an integer >= 1. "
            f"Provided value: {value!r}."
        )
    return int(value)


def _default_minimizer_settings(non_linearity, double_diode_updater):
    exp_clip = 100000.0
    if non_linearity == "double_diode_exponential":
        if double_diode_updater in ("float32", "overrelaxed"):
            exp_clip = 80.0
        elif double_diode_updater in ("float64_timed", "TimedExponentialDOubleDiodeUpdater"):
            exp_clip = 10000.0
    return MinimizerSettings(
        rel_tol=1e-5,
        vn_tol=1e-6,
        use_polish=True,
        max_newton_iters=32,
        z_thresh=1e10,
        exp_clip=exp_clip,
        dynamic_polish=True,
        overrelaxation_reject_steps=False,
        overrelaxation_reject_max_tries=3,
        overrelaxation_reject_shrink=0.5,
        overrelaxation_reject_eps=0.0,
    )


def _build_tracking_minimizer(fn, free_layers, model_cfg, mode, *, num_iterations, voltage_amp, current_amp):
    minimizer_cfg = model_cfg.get("minimizer", {})
    double_diode_updater = minimizer_cfg.get(
        "double_diode_updater", "CustomExponentialDoubleDiodeUpdater"
    )
    return TrackingQuadraticMinimizer(
        fn=fn,
        free_layers=free_layers,
        num_iterations=num_iterations,
        mode=mode,
        non_linearity=model_cfg["non_linearity"],
        quadratic_diode_param=model_cfg.get("quadratic_diode_param", {}),
        exponential_diode_param=model_cfg.get("exponential_diode_param", {}),
        hard_sigmoid_param=model_cfg.get("hard_sigmoid_param", {}),
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        iv_data=minimizer_cfg.get("iv_data"),
        iv_data_path=minimizer_cfg.get("iv_data_path"),
        double_diode_updater=double_diode_updater,
        adaptive_equilibrium=minimizer_cfg.get("adaptive_equilibrium", True),
        overrelaxation_factor=minimizer_cfg.get("overrelaxation_factor", 1.1),
        single_diode_updater=minimizer_cfg.get("single_diode_updater", "custom"),
        minimizer_settings=_default_minimizer_settings(
            model_cfg["non_linearity"], double_diode_updater
        ),
    )


def _single_conv_gradient_check(device):
    """Sanity check: ensure a standalone conv layer reaches quadratic equilibrium."""
    torch.manual_seed(0)

    batch_size = 2
    input_shape = (2, 6, 6)
    kernel = (3, 3)
    stride = 1
    padding = 0
    out_channels = 4
    h_out = (input_shape[1] + 2 * padding - kernel[0]) // stride + 1
    w_out = (input_shape[2] + 2 * padding - kernel[1]) // stride + 1
    output_shape = (out_channels, h_out, w_out)

    input_layer = ConvLayer(input_shape, device=device)
    output_layer = ConvLayer(output_shape, device=device)

    conv_weight = ConvWeight(
        shape=(out_channels, input_shape[0], kernel[0], kernel[1]),
        gain=0.5,
        device=device,
        clamp=False,
        clamp_min=None,
        clamp_max=None,
    )

    conv_interaction = ConvResistive(
        input_layer, output_layer, conv_weight, padding=padding, stride=stride, dilation=1
    )

    energy_fn = DetailedSumSeparableFunction(
        layers=[input_layer, output_layer],
        params=[conv_weight],
        interactions=[conv_interaction],
    )

    input_layer.state = torch.randn(batch_size, *input_shape, device=device)
    output_layer.state = torch.zeros(batch_size, *output_shape, device=device)

    minimizer = QuadraticMinimizer(
        energy_fn,
        free_layers=[output_layer],
        num_iterations=3,
        mode="forward",
        non_linearity="linear",
        quadratic_diode_param=_default_quadratic_params(),
        exponential_diode_param=_default_exponential_params(),
        voltage_amp=1.0,
        current_amp=1.0,
        hard_sigmoid_param={},
        iv_data=None,
        iv_data_path=None,
        double_diode_updater="CustomExponentialDoubleDiodeUpdater",
        adaptive_equilibrium=True,
        overrelaxation_factor=1.1,
        single_diode_updater="custom",
        minimizer_settings=_default_minimizer_settings("linear", "CustomExponentialDoubleDiodeUpdater"),
    )

    grad_fn = energy_fn.grad_layer_fn(output_layer)
    grad_before = grad_fn().norm().item()
    minimizer.compute_equilibrium()
    grad_after = grad_fn().norm().item()
    return grad_before, grad_after


def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = LABS_DIR / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return json.loads(config_path.read_text())


def _normalize_dataset_key(dataset_key):
    if dataset_key is None:
        return "mnist"
    normalized = str(dataset_key).strip().lower().replace("-", "_")
    if normalized == "fashion_mnist":
        return "fmnist"
    return normalized


def _resolve_dataset_config(config, dataset_key):
    normalized = _normalize_dataset_key(dataset_key)
    datasets_cfg = config.get("datasets", {})

    if normalized in datasets_cfg:
        return normalized, json.loads(json.dumps(datasets_cfg[normalized]))

    if normalized == "fmnist":
        base_cfg = datasets_cfg.get("mnist")
        if base_cfg is None:
            raise KeyError(
                "Config does not define datasets['mnist']; cannot derive Fashion-MNIST dataset settings."
            )

        dataset_cfg = json.loads(json.dumps(base_cfg))
        dataset_cfg["factory"] = "labs.datasets.FashionMnistDataset"

        params = dict(dataset_cfg.get("params", {}))
        params["name"] = "fmnist"
        root = params.get("root")
        if root is None or Path(str(root)).name == "mnist":
            params["root"] = "~/datasets/fashion_mnist"
        params.setdefault("download", True)
        params.setdefault("normalize", True)
        params.setdefault("normalize_mean", 0.286)
        params.setdefault("normalize_std", 0.353)
        dataset_cfg["params"] = params
        return normalized, dataset_cfg

    raise KeyError(f"Unsupported dataset key {dataset_key!r}.")


def _resolve_callable(import_path):
    module_path, attr = import_path.rsplit(".", 1)

    if module_path.startswith("labs."):
        rel_module = module_path[len("labs."):]
        module_base = rel_module.replace(".", "/")
        candidate = LABS_DIR / f"{module_base}.py"
        if not candidate.exists():
            candidate = LABS_DIR / rel_module.replace(".", "/") / "__init__.py"
        if not candidate.exists():
            raise ImportError(f"Cannot locate module '{module_path}' at {candidate}")

        spec = importlib.util.spec_from_file_location(
            f"labs_{rel_module.replace('.', '_')}", candidate
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = import_module(module_path)

    return getattr(module, attr)


def _resolve_run_dir(config_path: Path, output_dir: str | None) -> tuple[Path, str, str]:
    host = socket.gethostname().split(".")[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if output_dir:
        run_dir = Path(output_dir).expanduser().resolve()
    elif config_path.parent.name.startswith("trial_"):
        run_dir = config_path.parent
    else:
        run_dir = DEFAULT_IMAGE_RUNS_BASE / f"{config_path.stem}_{host}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, host, timestamp


def _resolve_initial_weights_path(value, config_path: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError(
            "Expected initial_weights to be null or a non-empty checkpoint path. "
            f"Provided value: {value!r}."
        )
    path = Path(value).expanduser()
    if path.is_absolute():
        path = path.resolve()
    else:
        cwd_path = (Path.cwd() / path).resolve()
        config_path_candidate = (config_path.parent / path).resolve()
        path = cwd_path if cwd_path.is_file() else config_path_candidate
    if not path.is_file():
        raise FileNotFoundError(
            "Expected initial_weights to name an existing model checkpoint. "
            f"Provided value: {str(path)!r}."
        )
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def _load_initial_parameters(energy_fn, checkpoint_path: Path, device) -> dict:
    states = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(states, (list, tuple)):
        raise ValueError(
            "Expected initial weights checkpoint to contain a list or tuple "
            "of parameter tensors. "
            f"Provided value: type={type(states).__name__!r}."
        )

    # Function.save() serializes the underlying ``_params`` list, while some
    # DRNs expose only trainable parameters from params() (for example, frozen
    # pooling conductances are omitted). Match the native checkpoint format.
    parameters = getattr(energy_fn, "_params", None)
    if parameters is None:
        parameters = energy_fn.params()
    parameters = list(parameters)
    if len(states) != len(parameters):
        raise ValueError(
            "Expected initial weights checkpoint to contain exactly "
            f"{len(parameters)} parameter tensors. "
            f"Provided value: {len(states)} tensors."
        )

    validated_states = []
    parameter_metadata = []
    for index, (parameter, state) in enumerate(zip(parameters, states)):
        if not torch.is_tensor(state) or not state.is_floating_point():
            raise ValueError(
                "Expected every initial parameter to be a floating-point tensor. "
                f"Provided value: index={index}, type={type(state).__name__!r}."
            )
        if tuple(state.shape) != tuple(parameter.state.shape):
            raise ValueError(
                "Expected every initial parameter shape to match the DRN. "
                f"Provided value: index={index}, checkpoint_shape={tuple(state.shape)!r}, "
                f"model_shape={tuple(parameter.state.shape)!r}."
            )
        if not bool(torch.isfinite(state).all()):
            raise ValueError(
                "Expected every initial parameter tensor to contain only finite values. "
                f"Provided value: index={index}."
            )
        loaded = state.to(
            device=parameter.state.device,
            dtype=parameter.state.dtype,
        )
        minimum = float(loaded.min().item())
        maximum = float(loaded.max().item())
        lower_bound = getattr(parameter, "min_cond", None)
        if lower_bound is None and getattr(parameter, "_non_negative", False):
            lower_bound = 0.0
        upper_bound = getattr(parameter, "max_cond", None)
        if lower_bound is not None and math.isfinite(float(lower_bound)):
            if minimum < float(lower_bound):
                raise ValueError(
                    "Expected every initial parameter to lie inside the DRN "
                    "parameter bounds. "
                    f"Provided value: index={index}, minimum={minimum!r}, "
                    f"lower_bound={float(lower_bound)!r}."
                )
        if upper_bound is not None and math.isfinite(float(upper_bound)):
            if maximum > float(upper_bound):
                raise ValueError(
                    "Expected every initial parameter to lie inside the DRN "
                    "parameter bounds. "
                    f"Provided value: index={index}, maximum={maximum!r}, "
                    f"upper_bound={float(upper_bound)!r}."
                )
        validated_states.append(loaded)
        parameter_metadata.append(
            {
                "index": index,
                "type": type(parameter).__name__,
                "shape": list(parameter.state.shape),
                "checkpoint_dtype": str(state.dtype),
                "model_dtype": str(parameter.state.dtype),
                "minimum": minimum,
                "maximum": maximum,
            }
        )

    for parameter, loaded in zip(parameters, validated_states):
        parameter.state.copy_(loaded)

    return {
        "path": str(checkpoint_path),
        "sha256": _sha256_file(checkpoint_path),
        "parameters": parameter_metadata,
    }


def _tensor_error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    difference = actual.detach().to(dtype=torch.float64) - expected.detach().to(
        device=actual.device,
        dtype=torch.float64,
    )
    return {
        "max_abs": float(difference.abs().max().item()),
        "mean_abs": float(difference.abs().mean().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
    }


@torch.no_grad()
def _evaluate_image_loader(
    network,
    cost_fn,
    energy_minimizer,
    dataloader,
    device,
    *,
    max_batches,
    label,
):
    running_loss = 0.0
    correct = 0
    seen = 0
    batches = 0
    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError(
                "Expected each evaluation batch to contain images and labels. "
                f"Provided value: type={type(batch).__name__!r}."
            )
        images, labels = batch[:2]
        images = images.to(device)
        labels = labels.to(device)
        network.set_input(images, reset=True)
        energy_minimizer.compute_equilibrium()
        cost_fn.set_target(labels)
        batch_loss = cost_fn.eval().mean().item()
        batch_correct = int((~cost_fn.error_fn()).sum().item())
        running_loss += batch_loss * images.size(0)
        correct += batch_correct
        seen += images.size(0)
        batches += 1

    loss = running_loss / seen if seen else float("nan")
    accuracy = correct / seen if seen else float("nan")
    print(
        f"[{label}] examples={seen} batches={batches} "
        f"loss={loss:.6f} accuracy={accuracy * 100:.2f}%"
    )
    return {
        "loss": loss,
        "accuracy": accuracy,
        "examples": seen,
        "batches": batches,
    }


def _initialization_mapping_diagnostics(
    energy_parameters,
    loaded_states,
    optimizer,
) -> list[dict]:
    persistent_reader = getattr(optimizer, "aihwkit_slow_conductance", None)
    auxiliary_reader = getattr(optimizer, "auxiliary_state", None)
    diagnostics = []
    for index, (parameter, loaded_state) in enumerate(
        zip(energy_parameters, loaded_states)
    ):
        item = {
            "index": index,
            "type": type(parameter).__name__,
            "shape": list(parameter.state.shape),
            "source_minimum": float(loaded_state.min().item()),
            "source_maximum": float(loaded_state.max().item()),
            "visible_realization_error": _tensor_error_stats(
                parameter.state,
                loaded_state,
            ),
        }
        if callable(persistent_reader):
            programmed = persistent_reader(parameter.state, persistent=True)
            if programmed is not None:
                item["persistent_programming_error"] = _tensor_error_stats(
                    programmed,
                    loaded_state,
                )
        if callable(auxiliary_reader):
            auxiliary = auxiliary_reader(parameter.state)
            if auxiliary is not None:
                item["fast_state_max_abs"] = float(
                    auxiliary.detach().abs().max().item()
                )
        diagnostics.append(item)
    return diagnostics


def _train_image_task(
    config_path,
    epochs,
    lr,
    beta,
    log_interval,
    max_batches,
    dataset_key,
    model_key,
    image_shape,
    device=None,
    sanity_check=False,
    epoch_callback=None,
    output_dir=None,
    max_test_batches=None,
    initial_weights=None,
    seed=None,
):
    max_test_batches = _validate_optional_batch_limit(
        "max_test_batches",
        max_test_batches,
    )
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    initial_weights_path = _resolve_initial_weights_path(
        initial_weights if initial_weights is not None else config.get("initial_weights"),
        config_path,
    )
    seed = seed if seed is not None else config.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError(
                "Expected seed to be null or an integer. "
                f"Provided value: {seed!r}."
            )
        seed = int(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    dataset_key = _normalize_dataset_key(dataset_key)
    run_dir, host, timestamp = _resolve_run_dir(config_path, output_dir)
    config_snapshot_path = run_dir / "config.used.json"
    config_snapshot_path.write_text(json.dumps(config, indent=2))

    project_root = PROJECT_ROOT
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    requested_device = device if device is not None else config.get("device")
    if requested_device is not None:
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {requested_device!r}, but CUDA is not available.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if sanity_check:
        grad_before, grad_after = _single_conv_gradient_check(device)
        print(
            f"[Sanity] Single conv gradient norm {grad_before:.6e} -> {grad_after:.6e}"
        )
        if not grad_after < grad_before * 1e-3:
            raise RuntimeError(
                "Single conv sanity check failed: gradients did not decrease sufficiently."
            )

    model_overrides = config["model_overrides"]
    resolved_model_key = model_key
    if resolved_model_key not in model_overrides:
        lab_model_key = config.get("lab", {}).get("model_key")
        if lab_model_key in model_overrides:
            resolved_model_key = lab_model_key
        elif dataset_key in model_overrides:
            resolved_model_key = dataset_key
        elif len(model_overrides) == 1:
            resolved_model_key = next(iter(model_overrides))
        else:
            available = ", ".join(sorted(model_overrides))
            raise KeyError(
                f"Unknown model_key {model_key!r}; available model_overrides keys: {available}"
            )

    model_cfg = {**config["model_base"], **model_overrides[resolved_model_key]}
    dataset_key, dataset_cfg = _resolve_dataset_config(config, dataset_key)
    beta_value = beta
    if beta_value is None:
        beta_value = config.get("beta")
        if beta_value is None:
            beta_value = max(model_cfg.get("nudging", 0.0), 1e-3)
    beta_value = float(beta_value)

    height, width = image_shape

    raw_layer_shapes = model_cfg["layer_shapes"]
    layer_shapes = [
        tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
        for shape in raw_layer_shapes
    ]
    conv_pipeline = model_cfg.get("conv_pipeline", [])
    if conv_pipeline is None:
        conv_pipeline = []

    input_shape = layer_shapes[0]
    if len(input_shape) != 3:
        raise ValueError(
            f"Expected input layer shape with 3 dimensions (channels, H, W); got {input_shape}"
        )
    if input_shape[1:] != (height, width):
        raise ValueError(
            f"Input layer shape {input_shape} does not match dataset image size {(height, width)}."
        )

    output_shape = layer_shapes[-1]
    if len(output_shape) != 1:
        raise ValueError(
            f"Output layer shape must be 1-D (num_classes or 2*num_classes); got {output_shape}"
        )

    num_classes = 10

    print(
        f"Training model '{model_key}' on dataset '{dataset_key}' | layer_shapes={layer_shapes} "
        f"| num_classes={num_classes} | non_linearity={model_cfg['non_linearity']} "
        f"| beta={beta_value}"
    )

    energy_fn = FlexibleDeepResistiveEnergy(
        layer_shapes=layer_shapes,
        conv_pipeline=conv_pipeline,
        pooling_mode=model_cfg.get("pooling_mode"),
        weight_gains=model_cfg["weight_gains"],
        input_gain=model_cfg["input_gain"],
        non_linearity=model_cfg["non_linearity"],
        exponential_diode_param=model_cfg["exponential_diode_param"],
        quadratic_diode_param=model_cfg["quadratic_diode_param"],
        hard_sigmoid_param=model_cfg.get("hard_sigmoid_param", {}),
        voltage_amp=model_cfg["voltage_amp"],
        current_amp=model_cfg["current_amp"],
        weight_min=model_cfg["weight_min"],
        weight_max=model_cfg["weight_max"],
        input_mode=config.get("input_mode", "train"),
    )
    energy_fn.set_device(device)
    initialization_metadata = None
    loaded_parameter_states = None
    if initial_weights_path is not None:
        initialization_metadata = _load_initial_parameters(
            energy_fn,
            initial_weights_path,
            device,
        )
        loaded_parameter_states = [
            parameter.state.detach().clone()
            for parameter in energy_fn.params()
        ]

    network = Network(energy_fn)
    free_layers = network.free_layers()
    output_layer = energy_fn.layers()[-1]

    if len(output_layer.shape) != 1:
        raise ValueError("Image setup expects a vector output layer.")

    output_dim = output_layer.shape[0]

    if output_dim == num_classes:
        cost_fn = SquaredError(output_layer)
    elif output_dim == 2 * num_classes:
        cost_fn = SquaredErrorPairedOutputs(output_layer, num_classes=num_classes)
    else:
        raise ValueError(
            f"Unsupported output_dim={output_dim}; expected {num_classes} or {2 * num_classes}."
        )
    
    augmented_fn = AugmentedFunction(energy_fn, cost_fn)

    training_iterations = int(
        model_cfg.get("num_iterations_training", model_cfg["num_iterations_inference"])
    )
    inference_iterations = int(model_cfg["num_iterations_inference"])
    minimizer_training = _build_tracking_minimizer(
        augmented_fn,
        free_layers,
        model_cfg,
        config["energy_minimizer"]["mode"],
        num_iterations=training_iterations,
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
    )

    minimizer_inference = _build_tracking_minimizer(
        energy_fn,
        free_layers,
        model_cfg,
        config["energy_minimizer"]["mode"],
        num_iterations=inference_iterations,
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
    )



    dataset_factory = _resolve_callable(dataset_cfg["factory"])
    dataset_params = dict(dataset_cfg["params"])
    dataset_params.setdefault("device", device)
    if "root" in dataset_params:
        dataset_params["root"] = os.path.expanduser(str(dataset_params["root"]))
    loader_result = dataset_factory(**dataset_params).build()
    if isinstance(loader_result, tuple):
        train_loader, test_loader = loader_result
    else:
        train_loader = loader_result
        test_loader = None

    if loaded_parameter_states is not None and test_loader is not None:
        initialization_metadata["loaded_fp32_evaluation"] = _evaluate_image_loader(
            network,
            cost_fn,
            minimizer_inference,
            test_loader,
            device,
            max_batches=max_test_batches,
            label="Initial loaded FP32",
        )

    params = augmented_fn.params()
    estimator = EquilibriumProp(
        params,
        free_layers,
        augmented_fn,
        cost_fn,
        minimizer_training,
        variant="centered",
        nudging=beta_value,
    )

    energy_params = energy_fn.params()
    cost_params = cost_fn.params()
    if isinstance(lr, (list, tuple)):
        if len(lr) < len(energy_params) + len(cost_params):
            raise ValueError(
                f"Expected {len(energy_params) + len(cost_params)} learning rates, got {len(lr)}."
            )
        learning_rates = list(lr)
    else:
        learning_rates = [lr] * (len(energy_params) + len(cost_params))
    optimizer = build_optimizer(
        energy_fn,
        cost_fn,
        learning_rates,
        update_pipeline=config.get("update_pipeline"),
        momentum=0.0,
        weight_decay=0.0,
    )
    if loaded_parameter_states is not None:
        initialization_metadata["mapping_diagnostics"] = (
            _initialization_mapping_diagnostics(
                energy_params,
                loaded_parameter_states,
                optimizer,
            )
        )
        if test_loader is not None:
            initialization_metadata["realized_aihwkit_evaluation"] = (
                _evaluate_image_loader(
                    network,
                    cost_fn,
                    minimizer_inference,
                    test_loader,
                    device,
                    max_batches=max_test_batches,
                    label="Initial realized AIHWKit",
                )
            )

    history = {
        "loss": [],
        "accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }
    if initialization_metadata is not None:
        history["initialization"] = initialization_metadata
    writer = SummaryWriter(str(run_dir)) if SummaryWriter is not None else None
    if writer is None:
        print("[mnist_train] TensorBoard unavailable; proceeding without event files.")
    else:
        print(f"[mnist_train] Writing TensorBoard events to {run_dir}")

    stop_training = False
    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        seen = 0
        reset_flag = True

        for batch_idx, (images, labels) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)

            network.set_input(images, reset=reset_flag)
            reset_flag = False
            minimizer_inference.compute_equilibrium()
            cost_fn.set_target(labels)

            batch_loss = cost_fn.eval().mean().item()
            errors = cost_fn.error_fn()
            batch_correct = int((~errors).sum().item())

            running_loss += batch_loss * images.size(0)
            running_correct += batch_correct
            seen += images.size(0)

            grads = estimator.compute_gradient()
            for param, grad in zip(params, grads[:len(params)]):
                param.state.grad = grad
            optimizer.step()
            for param in energy_params:
                param.clamp_()

            acc = running_correct / seen
            avg_loss = running_loss / seen
            print(
                f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                f"running loss={avg_loss:.4f} acc={acc*100:.2f}%",
                end="\r",
                flush=True,
            )

        if seen:
            epoch_train_loss = running_loss / seen
            epoch_train_acc = running_correct / seen
        else:
            epoch_train_loss = float("nan")
            epoch_train_acc = float("nan")
        history["loss"].append(epoch_train_loss)
        history["accuracy"].append(epoch_train_acc)

        if test_loader is not None:
            test_running_loss = 0.0
            test_correct = 0
            test_seen = 0
            optimizer.zero_grad()
            for test_batch_idx, (images, labels) in enumerate(test_loader):
                if max_test_batches is not None and test_batch_idx >= max_test_batches:
                    break
                images = images.to(device)
                labels = labels.to(device)
                network.set_input(images, reset=True)
                minimizer_inference.compute_equilibrium()
                cost_fn.set_target(labels)
                batch_loss = cost_fn.eval().mean().item()
                errors = cost_fn.error_fn()
                batch_correct = int((~errors).sum().item())
                test_running_loss += batch_loss * images.size(0)
                test_correct += batch_correct
                test_seen += images.size(0)

                avg_loss = test_running_loss / test_seen
                acc = test_correct / test_seen
                print(
                    f"[Test] Epoch {epoch+1}/{epochs} | Batch {test_batch_idx+1}/{len(test_loader)} | running loss={avg_loss:.4f} acc={acc*100:.2f}%",
                    end="\r",
                    flush=True,
                )
            epoch_test_loss = test_running_loss / test_seen if test_seen else float("nan")
            epoch_test_acc = test_correct / test_seen if test_seen else float("nan")
            history["test_loss"].append(epoch_test_loss)
            history["test_accuracy"].append(epoch_test_acc)
        else:
            epoch_test_loss = None
            epoch_test_acc = None

        print()

        if writer is not None:
            writer.add_scalar("train/loss", epoch_train_loss, epoch + 1)
            writer.add_scalar("train/accuracy", epoch_train_acc, epoch + 1)
            if epoch_test_loss is not None:
                writer.add_scalar("test/loss", epoch_test_loss, epoch + 1)
            if epoch_test_acc is not None:
                writer.add_scalar("test/accuracy", epoch_test_acc, epoch + 1)
                writer.add_scalar("test/error", 1.0 - epoch_test_acc, epoch + 1)
            writer.flush()

        if epoch_callback:
            epoch_info = {
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "train_accuracy": epoch_train_acc,
                "test_loss": epoch_test_loss,
                "test_accuracy": epoch_test_acc,
            }
            should_stop = epoch_callback(epoch_info, history)
            if should_stop:
                stop_training = True
                break

    if stop_training:
        print("Epoch callback requested early stop; summarizing current history.")

    def _last(values):
        return values[-1] if values else float("nan")

    summary = {
        "final_train_loss": _last(history["loss"]),
        "final_train_accuracy": _last(history["accuracy"]),
        "best_train_accuracy": max(history["accuracy"]) if history["accuracy"] else float("nan"),
    }
    if history["test_accuracy"]:
        summary["final_test_accuracy"] = history["test_accuracy"][-1]
        summary["best_test_accuracy"] = max(history["test_accuracy"])
        summary["final_test_loss"] = history["test_loss"][-1]
        summary["test_error"] = 1.0 - summary["final_test_accuracy"]
    else:
        summary["test_error"] = 1.0 - summary["final_train_accuracy"]
    if initialization_metadata is not None:
        loaded_evaluation = initialization_metadata.get(
            "loaded_fp32_evaluation"
        )
        realized_evaluation = initialization_metadata.get(
            "realized_aihwkit_evaluation"
        )
        if loaded_evaluation is not None:
            summary["initial_loaded_test_accuracy"] = loaded_evaluation[
                "accuracy"
            ]
            summary["initial_loaded_test_loss"] = loaded_evaluation["loss"]
        if realized_evaluation is not None:
            summary["initial_realized_test_accuracy"] = realized_evaluation[
                "accuracy"
            ]
            summary["initial_realized_test_loss"] = realized_evaluation["loss"]

    if writer is not None:
        writer.close()

    model_path = run_dir / "model.pt"
    energy_fn.save(str(model_path))
    event_files = sorted(str(path) for path in run_dir.glob("events.out.tfevents.*"))
    metadata = {
        "config_path": str(config_path),
        "config_used_path": str(config_snapshot_path),
        "run_dir": str(run_dir),
        "event_files": event_files,
        "epochs": int(epochs),
        "lr": list(lr) if isinstance(lr, (list, tuple)) else float(lr),
        "beta": beta_value,
        "max_batches": max_batches,
        "max_test_batches": max_test_batches,
        "dataset_key": dataset_key,
        "model_key": model_key,
        "dataset_factory": dataset_cfg["factory"],
        "device": str(device),
        "seed": seed,
        "host": host,
        "timestamp": timestamp,
        "initialization": initialization_metadata,
        "model_path": str(model_path),
        "summary": summary,
    }
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    summary["run_dir"] = str(run_dir)
    summary["event_files"] = event_files
    summary["config_used_path"] = str(config_snapshot_path)
    summary["run_metadata_path"] = str(metadata_path)
    summary["model_path"] = str(model_path)
    history["summary"] = summary
    history["run_dir"] = str(run_dir)
    history["event_files"] = event_files
    return history





def train_mnist_conv(
    config_path,
    epochs,
    lr,
    beta,
    log_interval,
    max_batches,
    device=None,
    sanity_check=False,
    epoch_callback=None,
    dataset_key="mnist",
    output_dir=None,
    model_key="mnist",
    max_test_batches=None,
    initial_weights=None,
    seed=None,
):
    return _train_image_task(
        config_path=config_path,
        epochs=epochs,
        lr=lr,
        beta=beta,
        log_interval=log_interval,
        max_batches=max_batches,
        dataset_key=dataset_key,
        model_key=model_key,
        image_shape=(28, 28),
        device=device,
        sanity_check=sanity_check,
        epoch_callback=epoch_callback,
        output_dir=output_dir,
        max_test_batches=max_test_batches,
        initial_weights=initial_weights,
        seed=seed,
    )


def train_tiny_grid(
    config_path,
    epochs,
    lr,
    beta,
    log_interval,
    max_batches,
    device=None,
    sanity_check=False,
    epoch_callback=None,
    output_dir=None,
    max_test_batches=None,
    initial_weights=None,
    seed=None,
):
    return _train_image_task(
        config_path=config_path,
        epochs=epochs,
        lr=lr,
        beta=beta,
        log_interval=log_interval,
        max_batches=max_batches,
        dataset_key="tiny3x3",
        model_key="tiny3x3",
        image_shape=(3, 3),
        device=device,
        sanity_check=sanity_check,
        epoch_callback=epoch_callback,
        output_dir=output_dir,
        max_test_batches=max_test_batches,
        initial_weights=initial_weights,
        seed=seed,
    )


def main():
    parser = argparse.ArgumentParser(description="Train resistive MNIST model via equilibrium propagation.")
    parser.add_argument("--config", required=True, help="Path to configuration JSON.")
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs. Overrides config if provided.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate for SGD updates. Overrides config if provided.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        help="Nudging strength for equilibrium propagation. Overrides config if provided.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        help="Logging interval in batches. Overrides config if provided.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Optional cap on number of batches per epoch. Overrides config if provided.",
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        help="Optional cap on number of test batches per epoch. Overrides config if provided.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Torch device string to use, e.g. 'cpu', 'cuda', or 'cuda:0'. Overrides config if provided.",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run internal single-conv gradient sanity check before training.",
    )
    parser.add_argument(
        "--result-json",
        type=str,
        default=None,
        help="Optional path to write a JSON summary (e.g., for Optuna sweeps).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory for run artifacts such as config.used.json, run_metadata.json, and TensorBoard events.",
    )
    parser.add_argument(
        "--initial-weights",
        type=str,
        default=None,
        help=(
            "Optional saved DRN model.pt used to initialize parameters before "
            "constructing the optimizer. Overrides config initial_weights."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional Python/NumPy/PyTorch seed. Overrides config seed.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset override for MNIST-style image runs. Use 'mnist' or 'fmnist'.",
    )

    args = parser.parse_args()
    provided_config_path = Path(args.config).expanduser()
    if provided_config_path.is_absolute() or provided_config_path.exists():
        config_path = provided_config_path.resolve()
    else:
        config_path = (LABS_DIR / provided_config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            "Expected --config to name an existing JSON file, either relative "
            "to the current directory or to labs/. "
            f"Provided value: {args.config!r}."
        )
    config = load_config(config_path)
    lab_cfg = config.get("lab", {})

    epochs = args.epochs if args.epochs is not None else lab_cfg.get("epochs")
    if epochs is None:
        raise ValueError("Epochs must be provided via --epochs or config['lab']['epochs'].")
    epochs = int(epochs)

    if args.lr is not None:
        lr = args.lr
    else:
        lr = config.get("lr")

    if lr is None:
        raise ValueError("Learning rate must be provided via --lr or config['lr'].")

    if isinstance(lr, list):
        learning_rate_cfg = [float(x) for x in lr]
    else:
        learning_rate_cfg = float(lr)

    beta = args.beta if args.beta is not None else config.get("beta")
    if beta is not None:
        beta = float(beta)

    log_interval = args.log_interval if args.log_interval is not None else config.get("log_interval")
    if log_interval is None:
        raise ValueError("Log interval must be provided via --log-interval or config['log_interval'].")
    log_interval = int(log_interval)

    max_batches = args.max_batches if args.max_batches is not None else config.get("max_batches")
    if max_batches is not None:
        max_batches = int(max_batches)

    max_test_batches = (
        args.max_test_batches
        if args.max_test_batches is not None
        else config.get("max_test_batches")
    )
    max_test_batches = _validate_optional_batch_limit(
        "max_test_batches",
        max_test_batches,
    )

    model_key = lab_cfg.get("model_key", "mnist")
    default_dataset_key = lab_cfg.get("dataset_key")
    if default_dataset_key is None:
        default_dataset_key = "tiny3x3" if model_key == "tiny3x3" else "mnist"
    dataset_key = _normalize_dataset_key(
        args.dataset if args.dataset is not None else default_dataset_key
    )

    if model_key == "tiny3x3":
        train_fn = train_tiny_grid
        if dataset_key != "tiny3x3":
            raise ValueError(
                f"Model {model_key!r} only supports dataset 'tiny3x3', got {dataset_key!r}."
            )
    else:
        train_fn = train_mnist_conv
        if dataset_key not in ("mnist", "fmnist"):
            raise ValueError(
                f"Unsupported dataset {dataset_key!r}. Use 'mnist' or 'fmnist'."
            )

    train_kwargs = dict(
        config_path=str(config_path),
        epochs=epochs,
        lr=learning_rate_cfg,
        beta=beta,
        log_interval=log_interval,
        max_batches=max_batches,
        max_test_batches=max_test_batches,
        device=args.device,
        sanity_check=args.sanity_check,
        output_dir=args.output_dir,
        initial_weights=args.initial_weights,
        seed=args.seed,
    )
    if model_key != "tiny3x3":
        train_kwargs["dataset_key"] = dataset_key
        train_kwargs["model_key"] = model_key

    history = train_fn(**train_kwargs)
    print("Training history:", history)

    summary = history.get("summary", {})
    print("Final summary:", json.dumps(summary, indent=2))

    if args.result_json:
        result_payload = {
            "config_path": str(config_path),
            "epochs": epochs,
            "learning_rate": learning_rate_cfg,
            "beta": beta,
            "max_batches": max_batches,
            "max_test_batches": max_test_batches,
            "seed": args.seed if args.seed is not None else config.get("seed"),
            "model_key": lab_cfg.get("model_key", "mnist"),
            "dataset_key": dataset_key,
            "test_error": summary.get("test_error"),
            "final_train_accuracy": summary.get("final_train_accuracy"),
            "final_test_accuracy": summary.get("final_test_accuracy"),
            "best_test_accuracy": summary.get("best_test_accuracy"),
            "initial_loaded_test_accuracy": summary.get(
                "initial_loaded_test_accuracy"
            ),
            "initial_realized_test_accuracy": summary.get(
                "initial_realized_test_accuracy"
            ),
            "run_dir": summary.get("run_dir"),
            "event_files": summary.get("event_files"),
            "model_path": summary.get("model_path"),
        }
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result_payload, indent=2))
        print(f"Wrote Optuna summary to {result_path}")


if __name__ == "__main__":
    main()
