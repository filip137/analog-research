#!/usr/bin/env python3
import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labs.datasets import MoonsDataset  # noqa: E402
from model.resistive.network import DeepResistiveEnergy  # noqa: E402
from model.function.network import Network  # noqa: E402
from labs.custom_minimizer import CustomQuadraticMinimizer as QuadraticMinimizer  # noqa: E402
from training.engine import ExperimentComponents  # noqa: E402
from training.epoch import Trainer  # noqa: E402
from training.optimizers import SGDOptimizer  # noqa: E402
from training.sgd import AugmentedFunction, EquilibriumProp  # noqa: E402
from model.function.cost import SquaredError, SquaredErrorPairedOutputs  # noqa: E402
from model.variable.parameter import Bias  # noqa: E402
from training.statistics import (  # noqa: E402
    Counter,
    CostStat,
    EnergyStat,
    ErrorStat,
    TopFiveErrorStat,
)


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_one(
    output_root: Path,
    hidden_dim: int,
    hidden_layers: int,
    *,
    num_epochs: int,
    num_iterations: int,
    base_lr: float,
    num_points: int,
    batch_size: int,
    seed: int | None,
) -> Path:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = 2
    output_dim = 4
    hidden_dims = [hidden_dim] * hidden_layers
    layer_shapes = [(input_dim * 2,)] + [(dim,) for dim in hidden_dims] + [(output_dim,)]
    weight_gains = [1] * (len(layer_shapes) - 1)
    quadratic_params = {"diode_conductance": 100.0, "v_min": -0.5, "v_max": 0.5}
    exponential_params = {"I_s": 1e-6, "V_t": 0.05, "V_off": 0.5}

    energy_fn = DeepResistiveEnergy(
        layer_shapes=layer_shapes,
        weight_gains=weight_gains,
        input_gain=10.0,
        non_linearity="double_diode_exponential",
        exponential_diode_param=exponential_params,
        quadratic_diode_param=quadratic_params,
        voltage_amp=1.0,
        current_amp=1.0,
        weight_min=1e-9,
        weight_max=100.0,
    )
    energy_fn.set_device(device)

    network = Network(energy_fn)
    output_layer = energy_fn.layers()[-1]
    if output_layer.shape[0] == 2:
        cost_fn = SquaredError(output_layer)
    else:
        cost_fn = SquaredErrorPairedOutputs(output_layer, 2)

    train_loader, _ = MoonsDataset(
        name="moons",
        batch_size=batch_size,
        device=device,
        num_samples=num_points,
    ).build()

    augmented_fn = AugmentedFunction(energy_fn, cost_fn)
    energy_minimizer = QuadraticMinimizer(
        fn=energy_fn,
        free_layers=network.free_layers(),
        num_iterations=num_iterations,
        mode="asynchronous",
        non_linearity="double_diode_quadratic",
        quadratic_diode_param=quadratic_params,
        exponential_diode_param=exponential_params,
        voltage_amp=1.0,
        current_amp=1.0,
    )

    params = energy_fn.params()
    estimator = EquilibriumProp(
        params,
        network.free_layers(),
        augmented_fn,
        cost_fn,
        energy_minimizer,
        variant="centered",
        nudging=1e-3,
    )

    lr = base_lr * hidden_layers
    learning_rates = []
    for param in params:
        if isinstance(param, Bias):
            learning_rates.append(0.0)
        else:
            learning_rates.append(lr)
    optimizer = SGDOptimizer(
        energy_fn,
        cost_fn,
        learning_rates,
        momentum=0.0,
        weight_decay=0.0,
    )

    trainer = Trainer(
        ExperimentComponents(
            network=network, cost_fn=cost_fn, energy_minimizer=energy_minimizer,
            parameters=tuple(params) + tuple(cost_fn.params()),
            differentiator=estimator, optimizer=optimizer,
        ),
        train_loader,
        reset_input=True,
    )
    train_counter = Counter(energy_fn, len(train_loader.dataset))
    train_counter_post = Counter(energy_fn, len(train_loader.dataset))
    train_counter_post.display = False

    trainer.add_statistic(train_counter, phase="free")
    trainer.add_statistic(train_counter_post, phase="gradients")
    trainer.add_statistic(EnergyStat(energy_fn), phase="free")
    trainer.add_statistic(CostStat(cost_fn), phase="free")
    error_stat = ErrorStat(cost_fn)
    trainer.add_statistic(error_stat, phase="free")
    if output_layer.shape[0] >= 5:
        trainer.add_statistic(TopFiveErrorStat(cost_fn), phase="free")

    run_dir = output_root / f"h{hidden_layers}_d{hidden_dim}" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        print(f"[train] h={hidden_layers} d={hidden_dim} epoch {epoch}/{num_epochs}")
        trainer.run(verbose=True)
        error_pct = error_stat.get()
        accuracy = 100.0 - error_pct
        print(f"[train] accuracy={accuracy:.3f}% (error={error_pct:.3f}%)")

    model_path = run_dir / "model.pt"
    energy_fn.save(model_path)

    metadata = {
        "hidden_dim": hidden_dim,
        "hidden_layers": hidden_layers,
        "num_epochs": num_epochs,
        "num_iterations": num_iterations,
        "num_points": num_points,
        "batch_size": batch_size,
        "base_lr": base_lr,
        "lr_used": lr,
        "device": str(device),
        "model_path": str(model_path),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train moons models across hidden sizes/layers and save metadata+model.",
    )
    parser.add_argument(
        "--output-root",
        default="/home/filip/paper_simulation_results_/moons_training_sweep",
    )
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[5, 10, 25, 50, 100])
    parser.add_argument("--hidden-layers", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--num-iterations", type=int, default=8)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--num-points", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for layers in args.hidden_layers:
        for dim in args.hidden_dims:
            _train_one(
                output_root,
                hidden_dim=dim,
                hidden_layers=layers,
                num_epochs=args.num_epochs,
                num_iterations=args.num_iterations,
                base_lr=args.base_lr,
                num_points=args.num_points,
                batch_size=args.batch_size,
                seed=args.seed,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
