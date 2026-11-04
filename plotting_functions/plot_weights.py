#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import json
import torch

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from networks.resistive.network import DeepResistiveEnergy

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def plot_weights():
    print("Loading DRN model...")
    
    # Load config
    config = load_config()
    model = "drn-xs"
    model_cfg = config["models"][model]
    
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
    
    # Get weight parameters
    weight_params = [param for param in energy_fn.params() if "Weight" in param.name]
    
    # Create a single figure with subplots for both weights
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DRN Weight Visualizations', fontsize=16)
    
    for i, weight_param in enumerate(weight_params):
        weight = weight_param.state.detach().cpu().numpy()
        
        if i == 0:  # First weight matrix
            # Weight 0: (2, 28, 28, 100) - reshape to 2D for visualization
            weight_reshaped = weight.reshape(-1, weight.shape[-1])  # (1568, 100)
            
            # Heatmap
            im1 = axes[0,0].imshow(weight_reshaped, cmap='viridis', aspect='auto')
            axes[0,0].set_title(f'Weight 0: Input→Hidden ({weight.shape})')
            axes[0,0].set_xlabel('Hidden Units')
            axes[0,0].set_ylabel('Input Units (flattened)')
            plt.colorbar(im1, ax=axes[0,0])
            
            # Distribution
            weight_flat = weight.flatten()
            axes[0,1].hist(weight_flat, bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[0,1].set_title(f'Weight 0 Distribution\nMin: {weight.min():.2e}, Max: {weight.max():.2e}')
            axes[0,1].set_xlabel('Weight Value')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].axvline(weight_flat.mean(), color='red', linestyle='--', 
                            label=f'Mean: {weight_flat.mean():.2e}')
            axes[0,1].legend()
            axes[0,1].set_yscale('log')
            
        else:  # Second weight matrix
            # Weight 1: (100, 10)
            
            # Heatmap
            im2 = axes[1,0].imshow(weight, cmap='viridis', aspect='auto')
            axes[1,0].set_title(f'Weight 1: Hidden→Output ({weight.shape})')
            axes[1,0].set_xlabel('Output Units')
            axes[1,0].set_ylabel('Hidden Units')
            plt.colorbar(im2, ax=axes[1,0])
            
            # Distribution
            weight_flat = weight.flatten()
            axes[1,1].hist(weight_flat, bins=50, alpha=0.7, color='green', edgecolor='black')
            axes[1,1].set_title(f'Weight 1 Distribution\nMin: {weight.min():.2e}, Max: {weight.max():.2e}')
            axes[1,1].set_xlabel('Weight Value')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].axvline(weight_flat.mean(), color='red', linestyle='--', 
                            label=f'Mean: {weight_flat.mean():.2e}')
            axes[1,1].legend()
            axes[1,1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    print("Weight visualizations displayed!")

if __name__ == "__main__":
    plot_weights()
