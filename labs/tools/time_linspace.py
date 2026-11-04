#!/usr/bin/env python3
import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path("/home/filip/server_code")
LABS_DIR = PROJECT_ROOT / "labs"
for path in (PROJECT_ROOT, LABS_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from networks.data.reference import LinSpaceDataset  # noqa: E402
from networks.resistive.network import DeepResistiveEnergy  # noqa: E402
from networks.function.network import Network  # noqa: E402
from networks.resistive.solver import CustomQuadraticMinimizer as QuadraticMinimizer  # noqa: E402


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_timed_linspace(
    hidden_dim: int,
    hidden_layers: int,
    num_iterations: int,
    *,
    input_dim: int = 2,
    output_dim: int = 4,
    linspace_min: float = -10.0,
    linspace_max: float = 10.0,
    linspace_samples: int = 30,
    batch_size: int = 10,
    voltage_amp: float = 1.0,
    current_amp: float = 1.0,
    non_linearity: str = "double_diode_exponential",
    seed: int | None = None,
    device: str | None = None,
    output_dir: Path | None = None,
) -> dict:
    _set_seed(seed)

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    hidden_dims = [hidden_dim] * hidden_layers
    layer_shapes = [(input_dim * 2,)] + [(dim,) for dim in hidden_dims] + [(output_dim,)]
    weight_gains = [1] * (len(layer_shapes) - 1)
    quadratic_params = {"diode_conductance": 100.0, "v_min": -0.5, "v_max": 0.5}
    exponential_params = {"I_s": 1e-6, "V_t": 0.05, "V_off": 0.5}

    energy_fn = DeepResistiveEnergy(
        layer_shapes=layer_shapes,
        weight_gains=weight_gains,
        input_gain=10.0,
        non_linearity=non_linearity,
        exponential_diode_param=exponential_params,
        quadratic_diode_param=quadratic_params,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        weight_min=1e-4,
        weight_max=1.0,
    )
    energy_fn.set_device(torch_device)

    network = Network(energy_fn)
    free_layers = network.free_layers()
    energy_minimizer = QuadraticMinimizer(
        fn=energy_fn,
        free_layers=free_layers,
        num_iterations=num_iterations,
        mode="asynchronous",
        non_linearity=non_linearity,
        quadratic_diode_param=quadratic_params,
        exponential_diode_param=exponential_params,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
    )

    grid_dataset = LinSpaceDataset(
        name="linspace_eval",
        batch_size=batch_size,
        device=torch_device,
        num_inputs=input_dim,
        min=linspace_min,
        max=linspace_max,
        num_samples=linspace_samples,
    )
    grid_loader = grid_dataset.build_mesh(
        xmin=linspace_min,
        xmax=linspace_max,
        ymin=linspace_min,
        ymax=linspace_max,
    )

    input_layer = network.layers()[0]
    original_gain = getattr(input_layer, "_gain", None)
    if original_gain is not None:
        input_layer._gain = 1.0

    if torch_device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for batch in grid_loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        network.set_input(x, reset=True)
        input_layer.set_input(x, mode="test")
        energy_minimizer.compute_equilibrium()
    if torch_device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if original_gain is not None:
        input_layer._gain = original_gain

    result = {
        "hidden_dim": hidden_dim,
        "hidden_layers": hidden_layers,
        "num_iterations": num_iterations,
        "linspace_samples": linspace_samples,
        "batch_size": batch_size,
        "voltage_amp": voltage_amp,
        "current_amp": current_amp,
        "device": str(torch_device),
        "elapsed_seconds": elapsed,
    }

    if output_dir is not None:
        output_dir = Path(output_dir).expanduser().resolve()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = output_dir / f"{timestamp}_timed_linspace"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "timing.json").write_text(json.dumps(result, indent=2))
        result["output_dir"] = str(run_dir)

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Time a linspace sweep with randomly initialized weights.",
    )
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--hidden-layers", type=int, required=True)
    parser.add_argument("--num-iterations", type=int, required=True)
    parser.add_argument("--linspace-samples", type=int, default=30)
    parser.add_argument("--linspace-min", type=float, default=-10.0)
    parser.add_argument("--linspace-max", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--voltage-amp", type=float, default=1.0)
    parser.add_argument("--current-amp", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    result = run_timed_linspace(
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        num_iterations=args.num_iterations,
        linspace_samples=args.linspace_samples,
        linspace_min=args.linspace_min,
        linspace_max=args.linspace_max,
        batch_size=args.batch_size,
        voltage_amp=args.voltage_amp,
        current_amp=args.current_amp,
        device=args.device,
        seed=args.seed,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
