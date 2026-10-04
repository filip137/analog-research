from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from training.lab.monitor import Monitor
from training.core.engine import ExperimentComponents
from training.lab.statistics import add_standard_statistics
from training.core.sgd import AugmentedFunction, EquilibriumProp
from training.core.optimizers import build_optimizer
from labs.common import MnistParts, CustomTrainer, build_evaluator


LABS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LABS_DIR.parent


def track_training_statistics(
    parts: MnistParts,
    num_iterations,
    weights_path,
    *,
    build_minimizer_fn,
    record_statistics=("calc_residual_current",),
    verbose=True,
    num_epochs: Optional[int] = None,
    output_dir: Optional[Path] = None,
):
    energy_fn = parts.energy_fn
    network = parts.network
    cost_fn = parts.cost_fn
    free_layers = parts.free_layers
    model_cfg = parts.model_cfg
    training_cfg = parts.training_cfg
    train_loader = parts.train_loader
    test_loader = parts.test_loader
    if weights_path:
        energy_fn.load(weights_path)
    energy_minimizer_inference = build_minimizer_fn(parts, num_iterations)
    augmented_fn = AugmentedFunction(energy_fn, cost_fn)
    energy_minimizer_training = build_minimizer_fn(parts, num_iterations, fn=augmented_fn)
    params = energy_fn.params()

    # EP defaults live in `EquilibriumProp` (variant='centered', nudging=0.25). Only pass overrides when present.
    nudging = model_cfg.get("nudging", training_cfg.get("nudging"))
    ep_variant = training_cfg.get("ep_variant")
    ep_kwargs = {}
    if nudging is not None:
        ep_kwargs["nudging"] = nudging
    if ep_variant is not None:
        ep_kwargs["variant"] = ep_variant
    estimator = EquilibriumProp(params, free_layers, augmented_fn, cost_fn, energy_minimizer_training, **ep_kwargs)

    learning_rates = parts.learning_rates
    if learning_rates is None:
        if "learning_rates" in model_cfg:
            learning_rates = list(model_cfg["learning_rates"])
        elif "learning_rates_biases" in model_cfg and "learning_rates_weights" in model_cfg:
            learning_rates = list(model_cfg["learning_rates_biases"]) + list(model_cfg["learning_rates_weights"])
        else:
            raise ValueError(
                "Missing learning rates: set `MnistParts.learning_rates` or provide "
                "`model_cfg['learning_rates']` (or `learning_rates_biases` + `learning_rates_weights`)."
            )

    # `training.core.optimizers.SGDOptimizer` filters out `PoolWeight` parameters (kept frozen), but configs may still provide
    # a learning-rate for them. If so, drop those entries to keep alignment with the optimizer's parameter list.
    from model.variable.parameter import PoolWeight

    full_params = energy_fn.params() + cost_fn.params()
    non_pool_mask = [not isinstance(p, PoolWeight) for p in full_params]
    expected = sum(non_pool_mask)
    if len(learning_rates) == len(full_params) and expected != len(full_params):
        learning_rates = [lr for lr, keep in zip(learning_rates, non_pool_mask) if keep]
    if len(learning_rates) != expected:
        raise ValueError(
            f"learning_rates length ({len(learning_rates)}) does not match optimizer parameter count ({expected}); "
            f"check model_cfg learning_rates_* vs energy_fn.params() (+ cost_fn.params())"
        )

    optimizer = build_optimizer(
        energy_fn,
        cost_fn,
        learning_rates,
        update_pipeline=training_cfg.get("update_pipeline"),
        momentum=training_cfg.get("momentum", 0.0),
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )
    scheduler_gamma = training_cfg.get("scheduler_gamma")
    if scheduler_gamma is None:
        raise ValueError("Missing `training.scheduler_gamma` for ExponentialLR scheduler.")
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)

    if output_dir is None:
        run_root = PROJECT_ROOT / "plots" / "mnist_training_stats"
        run_root.mkdir(parents=True, exist_ok=True)
        run_dir = run_root / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_iters_{num_iterations}"
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = Path(output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    trainer = CustomTrainer(
        ExperimentComponents(
            network=network, cost_fn=cost_fn,
            energy_minimizer=energy_minimizer_inference,
            parameters=tuple(params) + tuple(cost_fn.params()),
            differentiator=estimator, optimizer=optimizer,
        ),
        train_loader,
        reset_input=False,
        record_statistics=record_statistics,
    )
    evaluator = build_evaluator(network, cost_fn, test_loader, energy_minimizer_inference, model_cfg, record_statistics)
    quad_params = model_cfg.get("quadratic_diode_param", {})
    add_standard_statistics(
        trainer, energy_fn, cost_fn, training=True,
        non_linearity=model_cfg.get("non_linearity"),
        v_min=quad_params.get("v_min"), v_max=quad_params.get("v_max"),
    )
    monitor = Monitor(
        trainer=trainer, evaluator=evaluator, scheduler=scheduler,
        save_model=energy_fn.save, path=run_dir,
    )
    epochs = num_epochs if num_epochs is not None else training_cfg.get("num_epochs")
    if epochs is None:
        print("[track_training_statistics] missing training.num_epochs; defaulting to 1")
        epochs = 1
    monitor.run(epochs, verbose=verbose)
    test_error = monitor.test_error()
    print(f"[mnist_task] iterations={num_iterations} test_error={test_error:.4f} (saved to {run_dir})")
    parts.learning_rates = learning_rates
    parts.scheduler = scheduler
    # Residual currents are tracked per-layer per-batch by CustomTrainer/CustomEvaluator.
    return {
        "train": trainer.res_currents,
        "test": evaluator.res_currents,
    }
