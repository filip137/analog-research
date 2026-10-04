import argparse
import importlib.util
import json
import torch
import os
import sys
from pathlib import Path
from datetime import datetime
import socket

LABS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LABS_DIR.parent
DATASETS_MODULE_PATH = PROJECT_ROOT / "datasets.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labs.tests.network import DeepResistiveEnergy
from model.function.network import Network
from model.function.cost import SquaredError, SquaredErrorPairedOutputs
from model.variable.parameter import DenseWeight
from labs.custom_minimizer import CustomQuadraticMinimizer as QuadraticMinimizer
from training.core.sgd import EquilibriumProp, Backprop, AugmentedFunction
from training.lab.epoch import Trainer, Evaluator
from training.core.engine import ExperimentComponents, EvaluationComponents
from training.lab.statistics import add_standard_statistics
from training.core.optimizers import SGDOptimizer
from training.lab.monitor import Monitor


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


def load_config(config_path="config.json"):
    if not os.path.isabs(config_path):
        # resolve relative to drn_simple.py
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def _safe_path_component(value: str) -> str:
    """Normalize arbitrary values into filesystem-friendly folder names."""
    text = str(value).strip()
    if not text:
        return "unknown"
    safe = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in text)
    return safe.strip("._") or "unknown"


def _build_run_dir(base_path: Path, model: str, non_linearity: str, host: str, algorithm: str, timestamp: str) -> Path:
    """
    Build the run directory path.

    Runs are always grouped as:
    <base_path>/<model>/<non_linearity>/<host>_<model>_<algorithm>_<timestamp>
    """
    run_root = base_path / _safe_path_component(model) / _safe_path_component(non_linearity)
    run_name = f"{host}_{model}_{algorithm}_{timestamp}"
    return run_root / run_name



def main(argv=None):
    parser = argparse.ArgumentParser(description="Deep Resistive Networks")
    parser.add_argument("--model", type=str, default="drn-xs",
                        help="DRN architecture: drn-1h, drn-2h, drn-3h, drn-xs, drn-xl, drn-linear")
    parser.add_argument("--algorithm", type=str, default="EP",
                        help="Training algorithm: equilibrium propagation (EP) or backpropagation (BP)")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to configuration JSON file")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g., cpu, cuda, cuda:0). Defaults to auto-select CUDA if available.",
    )
    # --- runtime override for diode conductance ---
    parser.add_argument(
        "--diode_conductance",
        type=float,
        default=None,
        help="Override models[model].quadratic_diode_param.diode_conductance at runtime",
    )
    # --- runtime override for base path ---
    parser.add_argument("--base_path", type=str, default=None,
                        help="Override paths.base_path in config")
    parser.add_argument("--result_json", type=str, default=None,
                        help="Optional path to store JSON summary (e.g., final test error)")
    parser.add_argument("--weights", type=str, default=None,
                    help="Optional path to a saved model.pt to load before training/eval")
    parser.add_argument("--beta", type=float, default=None,
                    help="Overwrites beta")   
    parser.add_argument("--early_stop_error", type=float, default=None,
                        help="If set, stop run early when test error exceeds this threshold.")
    parser.add_argument("--early_stop_after", type=int, default=1,
                        help="Number of epochs to run before applying early-stop threshold (default: 1).")
    parser.add_argument("--early_stop_error2", type=float, default=None,
                        help="Optional second early-stop threshold.")
    parser.add_argument("--early_stop_after2", type=int, default=1,
                        help="Epoch index after which to apply the second threshold.")

    args = parser.parse_args(argv)

    # Load config
    config = load_config(args.config)
    model = args.model
    algorithm = args.algorithm
    base_path = args.base_path


    if model not in config["models"]:
        raise ValueError(f"Model {model} not found in config.json")

    model_cfg = config["models"][model]

    # Collect diode parameters (allow legacy flat keys as fallbacks)
    quadratic_params = dict(model_cfg.get("quadratic_diode_param", {}))
    hard_sigmoid_params = dict(model_cfg.get("hard_sigmoid_param", {}))
    # Fill hard-sigmoid defaults from quadratic params when missing.
    if "v_min" not in hard_sigmoid_params and "v_min" in quadratic_params:
        hard_sigmoid_params["v_min"] = quadratic_params["v_min"]
    if "v_max" not in hard_sigmoid_params and "v_max" in quadratic_params:
        hard_sigmoid_params["v_max"] = quadratic_params["v_max"]
    if "g_on" not in hard_sigmoid_params and "diode_conductance" in quadratic_params:
        hard_sigmoid_params["g_on"] = quadratic_params["diode_conductance"]
    if "g_off" not in hard_sigmoid_params and "g_on" in hard_sigmoid_params:
        hard_sigmoid_params["g_off"] = hard_sigmoid_params["g_on"]
    if "diode_conductance" in hard_sigmoid_params and "g_on" not in hard_sigmoid_params:
        hard_sigmoid_params["g_on"] = hard_sigmoid_params["diode_conductance"]
        hard_sigmoid_params.setdefault("g_off", hard_sigmoid_params["diode_conductance"])
    exponential_params = dict(model_cfg.get("exponential_diode_param", {}))

    if "diode_conductance" in model_cfg:
        quadratic_params.setdefault("diode_conductance", model_cfg["diode_conductance"])
    if "v_min" in model_cfg:
        quadratic_params.setdefault("v_min", model_cfg["v_min"])
    if "v_max" in model_cfg:
        quadratic_params.setdefault("v_max", model_cfg["v_max"])

    model_cfg["quadratic_diode_param"] = quadratic_params
    model_cfg["exponential_diode_param"] = exponential_params
    model_cfg["hard_sigmoid_param"] = hard_sigmoid_params

    # Allow runtime override for conductance
    if args.diode_conductance is not None:
        quadratic_params["diode_conductance"] = float(args.diode_conductance)
        print(f"[drn_config] Overriding models[{model}].quadratic_diode_param.diode_conductance -> {quadratic_params['diode_conductance']}")

    training_cfg = config["training"]

    # Hyperparameters from config
    layer_shapes = [tuple(s) for s in model_cfg["layer_shapes"]]
    conv_pipeline = model_cfg.get("conv_pipeline")
    pooling_mode = model_cfg.get("pooling_mode")
    weight_gains = model_cfg["weight_gains"]
    input_gain = model_cfg["input_gain"]
    num_iterations_inference = model_cfg["num_iterations_inference"]
    num_iterations_training = model_cfg["num_iterations_training"]
    nudging = model_cfg["nudging"]
    if args.beta is not None:
        nudging = args.beta
        model_cfg["nudging"] = nudging
    learning_rates_weights = model_cfg["learning_rates_weights"]
    learning_rates_biases = model_cfg["learning_rates_biases"]
    num_epochs = model_cfg["num_epochs"]
    early_stop_error = args.early_stop_error
    early_stop_after = args.early_stop_after or 1
    early_stop_error2 = args.early_stop_error2
    early_stop_after2 = args.early_stop_after2 or 1

    dataset = training_cfg["dataset"]
    batch_size = training_cfg["batch_size"]

    # Load data
    data_root = Path.home() / "data"
    print(f"Data is loaded from {data_root}")
    training_loader, test_loader = load_dataloaders(
        dataset, batch_size,
        augment_32x32=training_cfg.get("augment_32x32"),
        normalize=training_cfg.get("normalize"),
        normalize_std=training_cfg.get("normalize_std"),
        data_root=data_root,
    )

    # Build the network
    weight_min = model_cfg.get("weight_min")
    weight_max = model_cfg.get("weight_max")
    weight_init_mode = model_cfg.get("weight_init_mode", "kaiming_uniform")
    energy_fn = DeepResistiveEnergy(
        layer_shapes,
        weight_gains,
        input_gain,
        non_linearity=model_cfg["non_linearity"],
        exponential_diode_param=exponential_params,
        quadratic_diode_param=quadratic_params,
        voltage_amp=model_cfg["voltage_amp"],
        current_amp=model_cfg["current_amp"],
        weight_min=weight_min,
        weight_max=weight_max,
        weight_init_mode=weight_init_mode,
        conv_pipeline=conv_pipeline,
        pooling_mode=pooling_mode
    )

    # Select device
    if args.device:
        device = args.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"Requested CUDA device '{device}' but CUDA is not available")
        if device.startswith("cuda:"):
            device_idx = device.split("cuda:")[1]
            if device_idx.isdigit():
                device_idx_int = int(device_idx)
                if device_idx_int >= torch.cuda.device_count():
                    raise ValueError(f"Requested CUDA device index {device_idx_int} but only {torch.cuda.device_count()} device(s) available")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    energy_fn.set_device(device)
    if args.weights:
        energy_fn.load(args.weights)  # maps to current device via map_location

    # Cost function
    output_layer = energy_fn.layers()[-1]
    if output_layer.shape[0] == 10:
        cost_fn = SquaredError(output_layer)
    elif output_layer.shape[0] == 20:
        cost_fn = SquaredErrorPairedOutputs(output_layer, 10)
    else:
        raise ValueError(f"Bad output layer shape: {output_layer.shape}")

    network = Network(energy_fn)

    # Energy minimizer and gradient estimator
    params = energy_fn.params()
    layers = energy_fn.layers()
    free_layers = network.free_layers()

    minimizer_mode = config["energy_minimizer"]["mode"]

    if algorithm == "EP":
        augmented_fn = AugmentedFunction(energy_fn, cost_fn)
        energy_minimizer_training = QuadraticMinimizer(
            fn=augmented_fn,
            free_layers=free_layers,
            num_iterations=num_iterations_training,
            mode=minimizer_mode,
            non_linearity=model_cfg["non_linearity"],
            quadratic_diode_param=quadratic_params,
            exponential_diode_param=exponential_params,
            hard_sigmoid_param=hard_sigmoid_params,
            voltage_amp=model_cfg["voltage_amp"],
            current_amp=model_cfg["current_amp"],
        )
        estimator = EquilibriumProp(params, layers, augmented_fn, cost_fn, energy_minimizer_training)
        estimator.nudging = nudging
        estimator.variant = config["algorithms"]["EP"].get("variant", "centered")
    elif algorithm == "BP":
        energy_minimizer_training = QuadraticMinimizer(
            fn=energy_fn,
            free_layers=free_layers,
            num_iterations=num_iterations_training,
            mode=minimizer_mode,
            non_linearity=model_cfg["non_linearity"],
            quadratic_diode_param=quadratic_params,
            exponential_diode_param=exponential_params,
            hard_sigmoid_param=hard_sigmoid_params,
            voltage_amp=model_cfg["voltage_amp"],
            current_amp=model_cfg["current_amp"],
        )
        estimator = Backprop(params, layers, cost_fn, energy_minimizer_training)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    energy_minimizer_training.num_iterations = num_iterations_training
    energy_minimizer_training.mode = minimizer_mode

    # Optimizer
    learning_rates = learning_rates_weights + learning_rates_biases
    if args.weights:
        gamma = 1
        learning_rates = [lr*gamma for lr in learning_rates]
    optimizer = SGDOptimizer(
        energy_fn, cost_fn, learning_rates,
        training_cfg.get("momentum", 0.0),
        training_cfg.get("weight_decay", 0.0)
    )

    # Trainer & evaluator
    energy_minimizer_inference = QuadraticMinimizer(
        fn=energy_fn,
        free_layers=free_layers,
        num_iterations=num_iterations_training,
        mode=minimizer_mode,
        non_linearity=model_cfg["non_linearity"],
        quadratic_diode_param=quadratic_params,
        exponential_diode_param=exponential_params,
        hard_sigmoid_param=hard_sigmoid_params,
        voltage_amp=model_cfg["voltage_amp"],
        current_amp=model_cfg["current_amp"],
    )
    energy_minimizer_inference.num_iterations = num_iterations_inference
    energy_minimizer_inference.mode = minimizer_mode

    trainer = Trainer(
        ExperimentComponents(
            network, cost_fn, energy_minimizer_inference,
            tuple(params) + tuple(cost_fn.params()), estimator, optimizer,
        ),
        training_loader, reset_input=True,
    )
    evaluator = Evaluator(
        EvaluationComponents(network, cost_fn, energy_minimizer_inference),
        test_loader, reset_input=True,
    )
    for runner, training in ((trainer, True), (evaluator, False)):
        add_standard_statistics(runner, energy_fn, cost_fn, training=training)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=training_cfg.get("scheduler_gamma")
    )
 
    # Monitor
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    host = socket.gethostname().split('.')[0]  # e.g. 'velociraptor', 'integnano1', 'nom-cool-2'

    if base_path is None:
        base_path = str(LABS_DIR / "cases")
        base_path_path = Path(base_path)
    else:
        # Use the supplied root directly (for example from sweep tooling).
        base_path_path = Path(args.base_path)
    run_dir = _build_run_dir(
        base_path_path,
        model=model,
        non_linearity=model_cfg["non_linearity"],
        host=host,
        algorithm=algorithm,
        timestamp=timestamp,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist the resolved configuration for reproducibility.
    config_snapshot = json.loads(json.dumps(config))
    config_snapshot.setdefault("paths", {})["base_path"] = str(run_dir)
    snapshot_path = run_dir / "config.used.json"
    with snapshot_path.open("w") as fh:
        json.dump(config_snapshot, fh, indent=2)

    path = str(run_dir)

    monitor = Monitor(
        trainer=trainer, evaluator=evaluator, scheduler=scheduler,
        save_model=energy_fn.save, path=path,
    )

    # Print info
    print(f"Dataset: {dataset} -- batch_size={batch_size}")
    print("Network:", energy_fn)
    print("Cost function:", cost_fn)
    print("Energy minimizer during inference:", energy_minimizer_inference)
    print("Energy minimizer during training:", energy_minimizer_training)
    print("Gradient estimator:", estimator)
    print("Parameter optimizer:", optimizer)
    print("Number of epochs =", num_epochs)
    print("Path =", path)
    print("Device =", device)
    print()

    # Run training and stream test error every epoch
    for epoch_idx in range(num_epochs):
        monitor.run(1, verbose=config["debug"].get("verbose"))
        test_error = float(monitor.test_error())
        print(f"[epoch {epoch_idx + 1}/{num_epochs}] test error: {test_error:.4f}", flush=True)
        if early_stop_error is not None and (epoch_idx + 1) >= early_stop_after:
            if test_error > early_stop_error:
                print(f"Early stopping: test_error {test_error:.4f} exceeded threshold {early_stop_error} at epoch {epoch_idx + 1}")
                break
        if early_stop_error2 is not None and (epoch_idx + 1) >= early_stop_after2:
            if test_error > early_stop_error2:
                print(f"Early stopping: test_error {test_error:.4f} exceeded threshold {early_stop_error2} at epoch {epoch_idx + 1}")
                break

    test_error = float(monitor.test_error())
    print(f"Final test error: {test_error:.4f}")

    if args.result_json:
        result_payload = {
            "model": model,
            "algorithm": algorithm,
            "test_error": test_error,
            "config_overrides": {
                "weight_gains": model_cfg["weight_gains"],
                "input_gain": model_cfg["input_gain"],
                "nudging": model_cfg["nudging"],
                "learning_rates_weights": model_cfg["learning_rates_weights"],
                "learning_rates_biases": model_cfg["learning_rates_biases"],
            },
        }
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("w") as fh:
            json.dump(result_payload, fh, indent=2)


if __name__ == "__main__":

    main()
