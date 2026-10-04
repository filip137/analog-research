#!/usr/bin/env python3
import sys
import os
import json
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from model.resistive.network import DeepResistiveEnergy
from training.core.sgd import EquilibriumProp, AugmentedFunction
from model.function.cost import SquaredError, SquaredErrorPairedOutputs
from datasets import load_dataloaders

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def calculate_layer_differences(model="drn-xs", num_samples=10):
    """
    Calculate the average difference between layers_first and layers_second
    in Equilibrium Propagation for a given number of samples.
    """
    
    print(f"Calculating layer differences for model: {model}")
    print(f"Using {num_samples} samples")
    
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
    
    # Create Equilibrium Propagation estimator
    augmented_fn = AugmentedFunction(energy_fn, cost_fn)
    estimator = EquilibriumProp(augmented_fn, model_cfg["nudging"])
    
    # Storage for differences
    layer_differences = []
    sample_count = 0
    
    print("\nProcessing samples...")
    
    for batch_idx, (inputs, targets) in enumerate(training_loader):
        if sample_count >= num_samples:
            break
            
        print(f"  Sample {sample_count + 1}/{num_samples}")
        
        # Set inputs and targets
        energy_fn.set_input(inputs)
        cost_fn.set_target(targets)
        
        # Compute gradients (this gives us layers_first and layers_second)
        try:
            grads = estimator.compute_gradient()
            
            # Get the layer states from the estimator
            # We need to access the internal states from the two nudging phases
            layers_first = estimator._layers_first
            layers_second = estimator._layers_second
            
            # Calculate differences for each layer
            batch_differences = []
            for i, (layer_first, layer_second) in enumerate(zip(layers_first, layers_second)):
                # Calculate absolute difference
                diff = torch.abs(layer_first - layer_second)
                
                # Calculate statistics
                mean_diff = diff.mean().item()
                max_diff = diff.max().item()
                std_diff = diff.std().item()
                
                batch_differences.append({
                    'layer': i,
                    'mean_diff': mean_diff,
                    'max_diff': max_diff,
                    'std_diff': std_diff,
                    'shape': layer_first.shape
                })
                
                print(f"    Layer {i}: mean_diff={mean_diff:.6f}, max_diff={max_diff:.6f}, std_diff={std_diff:.6f}")
            
            layer_differences.append(batch_differences)
            sample_count += 1
            
        except Exception as e:
            print(f"    Error processing sample: {e}")
            continue
    
    # Calculate average differences across all samples
    print(f"\n=== AVERAGE LAYER DIFFERENCES ({sample_count} samples) ===")
    
    if layer_differences:
        num_layers = len(layer_differences[0])
        
        for layer_idx in range(num_layers):
            # Collect statistics for this layer across all samples
            mean_diffs = [sample[layer_idx]['mean_diff'] for sample in layer_differences]
            max_diffs = [sample[layer_idx]['max_diff'] for sample in layer_differences]
            std_diffs = [sample[layer_idx]['std_diff'] for sample in layer_differences]
            
            # Calculate averages
            avg_mean_diff = np.mean(mean_diffs)
            avg_max_diff = np.mean(max_diffs)
            avg_std_diff = np.mean(std_diffs)
            
            # Get layer shape
            layer_shape = layer_differences[0][layer_idx]['shape']
            
            print(f"Layer {layer_idx} ({layer_shape}):")
            print(f"  Average mean difference: {avg_mean_diff:.6f}")
            print(f"  Average max difference:  {avg_max_diff:.6f}")
            print(f"  Average std difference:  {avg_std_diff:.6f}")
            print(f"  Range of mean diffs:     [{np.min(mean_diffs):.6f}, {np.max(mean_diffs):.6f}]")
            print()
    
    return layer_differences

def analyze_nudging_impact(model="drn-xs", nudging_values=[0.1, 1.0, 10.0], num_samples=5):
    """
    Analyze how different nudging values affect the layer differences.
    """
    print(f"=== ANALYZING NUDGING IMPACT ===")
    
    config = load_config()
    model_cfg = config["models"][model]
    
    results = {}
    
    for nudging in nudging_values:
        print(f"\nTesting nudging = {nudging}")
        
        # Temporarily modify nudging in config
        original_nudging = model_cfg["nudging"]
        model_cfg["nudging"] = nudging
        
        try:
            differences = calculate_layer_differences(model, num_samples)
            results[nudging] = differences
        except Exception as e:
            print(f"Error with nudging {nudging}: {e}")
        
        # Restore original nudging
        model_cfg["nudging"] = original_nudging
    
    return results

if __name__ == "__main__":
    # Calculate layer differences for default configuration
    differences = calculate_layer_differences("drn-xs", num_samples=5)
    
    # Optional: Analyze nudging impact
    # nudging_results = analyze_nudging_impact("drn-xs", [0.1, 1.0, 10.0], num_samples=3)
