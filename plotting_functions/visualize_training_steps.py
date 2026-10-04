#!/usr/bin/env python3
import sys
import json
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add this checkout's project root to the path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.resistive.network import DeepResistiveEnergy
from labs.custom_minimizer import CustomQuadraticMinimizer as QuadraticMinimizer
from training.sgd import EquilibriumProp, AugmentedFunction
from model.function.cost import SquaredError, SquaredErrorPairedOutputs
from training.optimizers import build_optimizer
from datasets import load_dataloaders

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def visualize_training_step(model="drn-xs", batch_idx=0, verbose=True):
    """
    Visualize the detailed steps of one training iteration.
    """
    
    print(f"=== VISUALIZING TRAINING STEPS FOR MODEL: {model} ===")
    
    # Load config
    config = load_config()
    model_cfg = config["models"][model]
    training_cfg = config["training"]
    quadratic_params = dict(model_cfg["quadratic_diode_param"])
    hard_sigmoid_params = dict(model_cfg.get("hard_sigmoid_param", {}))
    exponential_params = dict(model_cfg["exponential_diode_param"])
    minimizer_mode = config["energy_minimizer"]["mode"]
    
    # Build the network
    layer_shapes = [tuple(s) for s in model_cfg["layer_shapes"]]
    weight_gains = model_cfg["weight_gains"]
    input_gain = model_cfg["input_gain"]
    
    energy_fn = DeepResistiveEnergy(
        layer_shapes, weight_gains, input_gain,
        non_linearity=model_cfg["non_linearity"],
        voltage_amp=model_cfg["voltage_amp"],
        current_amp=model_cfg["current_amp"]
    )
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    energy_fn.set_device(device)
    print(f"Using device: {device}")
    
    # Cost function
    output_layer = energy_fn.layers()[-1]
    if output_layer.shape[0] == 10:
        cost_fn = SquaredError(output_layer)
    elif output_layer.shape[0] == 20:
        cost_fn = SquaredErrorPairedOutputs(output_layer, 10)
    else:
        raise ValueError(f"Unexpected output layer size: {output_layer.shape[0]}")
    
    # Load data
    training_loader, test_loader = load_dataloaders(
        training_cfg["dataset"], training_cfg["batch_size"],
        augment_32x32=training_cfg.get("augment_32x32", False),
        normalize=training_cfg.get("normalize", False)
    )
    
    # Get a batch
    for i, (inputs, targets) in enumerate(training_loader):
        if i == batch_idx:
            break
    
    print(f"\n=== BATCH {batch_idx} ===")
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Target values: {targets[:5].tolist()}")  # Show first 5 targets
    
    # Set inputs and targets
    energy_fn.set_input(inputs)
    cost_fn.set_target(targets)
    
    # Create energy minimizer
    free_layers = energy_fn.layers()[1:]
    energy_minimizer = QuadraticMinimizer(
        fn=energy_fn,
        free_layers=free_layers,
        num_iterations=model_cfg["num_iterations_training"],
        mode=minimizer_mode,
        non_linearity=model_cfg["non_linearity"],
        quadratic_diode_param=quadratic_params,
        exponential_diode_param=exponential_params,
        hard_sigmoid_param=hard_sigmoid_params,
        voltage_amp=model_cfg["voltage_amp"],
        current_amp=model_cfg["current_amp"],
    )
    
    # Create Equilibrium Propagation estimator
    params = energy_fn.params()
    layers = energy_fn.layers()
    estimator = EquilibriumProp(
        params, layers, energy_fn, cost_fn, energy_minimizer,
        variant='centered', nudging=model_cfg["nudging"]
    )
    
    # Create optimizer
    learning_rates_weights = model_cfg["learning_rates_weights"]
    learning_rates_biases = model_cfg["learning_rates_biases"]
    learning_rates = learning_rates_biases + learning_rates_weights
    optimizer = build_optimizer(
        energy_fn, cost_fn, learning_rates,
        update_pipeline=training_cfg.get("update_pipeline"),
        momentum=training_cfg.get("momentum", 0.0),
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )
    
    print(f"\n=== STEP-BY-STEP TRAINING PROCESS ===")
    
    # Step 1: Initial state
    print("\n1. INITIAL STATE:")
    print_parameter_stats(params, "Before any computation")
    
    # Step 2: Free phase (equilibrium)
    print("\n2. FREE PHASE (Equilibrium):")
    print("   Computing equilibrium with nudging = 0...")
    energy_minimizer.compute_equilibrium()
    print_layer_stats(layers, "After free phase")
    
    # Step 3: Cost evaluation
    print("\n3. COST EVALUATION:")
    cost_value = cost_fn.eval().mean().item()
    print(f"   Cost value: {cost_value:.6f}")
    
    # Step 4: Gradient computation
    print("\n4. GRADIENT COMPUTATION:")
    print("   Computing gradients via Equilibrium Propagation...")
    grads = estimator.compute_gradient()
    print_gradient_stats(params, grads, "Computed gradients")
    
    # Step 5: Set gradients
    print("\n5. SETTING GRADIENTS:")
    for param, grad in zip(params, grads):
        param.state.grad = grad
    print("   Gradients set to parameter.grad")
    
    # Step 6: Weight update
    print("\n6. WEIGHT UPDATE:")
    print("   Before optimizer.step():")
    print_parameter_stats(params, "Before update")
    
    optimizer.step()
    
    print("   After optimizer.step():")
    print_parameter_stats(params, "After update")
    
    # Step 7: Parameter clamping
    print("\n7. PARAMETER CLAMPING:")
    print("   Before clamping:")
    print_parameter_stats(params, "Before clamping")
    
    for param in params:
        param.clamp_()
    
    print("   After clamping:")
    print_parameter_stats(params, "After clamping")
    
    print(f"\n=== TRAINING STEP COMPLETE ===")

def print_parameter_stats(params, stage):
    """Print statistics about parameters at a given stage"""
    print(f"   {stage}:")
    
    for i, param in enumerate(params):
        param_name = param.name
        param_state = param.state
        
        if param_state.grad is not None:
            grad_norm = param_state.grad.norm().item()
            grad_mean = param_state.grad.mean().item()
            print(f"     {param_name}: shape={param_state.shape}, "
                  f"mean={param_state.mean().item():.6f}, "
                  f"std={param_state.std().item():.6f}, "
                  f"grad_norm={grad_norm:.6f}, "
                  f"grad_mean={grad_mean:.6f}")
        else:
            print(f"     {param_name}: shape={param_state.shape}, "
                  f"mean={param_state.mean().item():.6f}, "
                  f"std={param_state.std().item():.6f}, "
                  f"grad=None")

def print_gradient_stats(params, grads, stage):
    """Print statistics about gradients"""
    print(f"   {stage}:")
    
    for i, (param, grad) in enumerate(zip(params, grads)):
        param_name = param.name
        grad_norm = grad.norm().item()
        grad_mean = grad.mean().item()
        grad_std = grad.std().item()
        print(f"     {param_name}: grad_norm={grad_norm:.6f}, "
              f"grad_mean={grad_mean:.6f}, "
              f"grad_std={grad_std:.6f}")

def print_layer_stats(layers, stage):
    """Print statistics about layer states"""
    print(f"   {stage}:")
    
    for i, layer in enumerate(layers):
        layer_state = layer.state
        print(f"     Layer {i} ({layer.name}): shape={layer_state.shape}, "
              f"mean={layer_state.mean().item():.6f}, "
              f"std={layer_state.std().item():.6f}")

def compare_weight_updates(model="drn-xs", num_batches=3):
    """
    Compare weight updates across multiple batches to see the learning progress.
    """
    
    print(f"=== COMPARING WEIGHT UPDATES ACROSS {num_batches} BATCHES ===")
    
    # Load config
    config = load_config()
    model_cfg = config["models"][model]
    training_cfg = config["training"]
    
    # Build the network
    layer_shapes = [tuple(s) for s in model_cfg["layer_shapes"]]
    weight_gains = model_cfg["weight_gains"]
    input_gain = model_cfg["input_gain"]
    
    energy_fn = DeepResistiveEnergy(
        layer_shapes, weight_gains, input_gain,
        non_linearity=model_cfg["non_linearity"],
        voltage_amp=model_cfg["voltage_amp"],
        current_amp=model_cfg["current_amp"]
    )
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    energy_fn.set_device(device)
    
    # Cost function
    output_layer = energy_fn.layers()[-1]
    if output_layer.shape[0] == 10:
        cost_fn = SquaredError(output_layer)
    elif output_layer.shape[0] == 20:
        cost_fn = SquaredErrorPairedOutputs(output_layer, 10)
    else:
        raise ValueError(f"Unexpected output layer size: {output_layer.shape[0]}")
    
    # Load data
    training_loader, test_loader = load_dataloaders(
        training_cfg["dataset"], training_cfg["batch_size"],
        augment_32x32=training_cfg.get("augment_32x32", False),
        normalize=training_cfg.get("normalize", False)
    )
    
    # Create energy minimizer
    free_layers = energy_fn.layers()[1:]
    energy_minimizer = QuadraticMinimizer(
        fn=energy_fn,
        free_layers=free_layers,
        num_iterations=model_cfg["num_iterations_training"],
        mode=minimizer_mode,
        non_linearity=model_cfg["non_linearity"],
        quadratic_diode_param=quadratic_params,
        exponential_diode_param=exponential_params,
        hard_sigmoid_param=hard_sigmoid_params,
        voltage_amp=model_cfg["voltage_amp"],
        current_amp=model_cfg["current_amp"],
    )
    
    # Create Equilibrium Propagation estimator
    params = energy_fn.params()
    layers = energy_fn.layers()
    estimator = EquilibriumProp(
        params, layers, energy_fn, cost_fn, energy_minimizer,
        variant='centered', nudging=model_cfg["nudging"]
    )
    
    # Create optimizer
    learning_rates_weights = model_cfg["learning_rates_weights"]
    learning_rates_biases = model_cfg["learning_rates_biases"]
    learning_rates = learning_rates_biases + learning_rates_weights
    optimizer = build_optimizer(
        energy_fn, cost_fn, learning_rates,
        update_pipeline=training_cfg.get("update_pipeline"),
        momentum=training_cfg.get("momentum", 0.0),
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )
    
    # Store initial weights
    initial_weights = [param.state.clone() for param in params]
    
    batch_count = 0
    for batch_idx, (inputs, targets) in enumerate(training_loader):
        if batch_count >= num_batches:
            break
            
        print(f"\n--- BATCH {batch_count} ---")
        
        # Set inputs and targets
        energy_fn.set_input(inputs)
        cost_fn.set_target(targets)
        
        # Compute gradients and update
        grads = estimator.compute_gradient()
        for param, grad in zip(params, grads):
            param.state.grad = grad
        
        # Store weights before update
        weights_before = [param.state.clone() for param in params]
        
        # Update weights
        optimizer.step()
        
        # Store weights after update
        weights_after = [param.state.clone() for param in params]
        
        # Calculate weight changes
        print("Weight changes:")
        for i, (param, w_before, w_after) in enumerate(zip(params, weights_before, weights_after)):
            weight_change = (w_after - w_before).abs().mean().item()
            print(f"  {param.name}: mean_abs_change = {weight_change:.8f}")
        
        # Clamp parameters
        for param in params:
            param.clamp_()
        
        batch_count += 1
    
    # Compare with initial weights
    print(f"\n=== TOTAL CHANGE FROM INITIAL WEIGHTS ===")
    for i, (param, initial_weight) in enumerate(zip(params, initial_weights)):
        total_change = (param.state - initial_weight).abs().mean().item()
        print(f"  {param.name}: total_mean_abs_change = {total_change:.8f}")

if __name__ == "__main__":
    # Visualize one training step in detail
    visualize_training_step("drn-xs", batch_idx=0, verbose=True)
    
    print("\n" + "="*80 + "\n")
    
    # Compare weight updates across multiple batches
    compare_weight_updates("drn-xs", num_batches=3)
