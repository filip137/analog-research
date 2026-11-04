#!/usr/bin/env python3
import sys
import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from networks.resistive.network import DeepResistiveEnergy

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def visualize_weights():
    print("Loading DRN model...")
    
    # Load config
    config = load_config()
    model = "drn-xs"  # Change this to any model you want
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
    
    print(f"Found {len(weight_params)} weight matrices")
    
    # Visualize each weight matrix
    for i, weight_param in enumerate(weight_params):
        weight = weight_param.state.detach().cpu().numpy()
        print(f"\nWeight {i} ({weight_param.name}):")
        print(f"  Shape: {weight.shape}")
        print(f"  Min: {weight.min():.6f}")
        print(f"  Max: {weight.max():.6f}")
        print(f"  Mean: {weight.mean():.6f}")
        print(f"  Std: {weight.std():.6f}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Weight Matrix {i} ({weight_param.name})', fontsize=16)
        
        # 1. Weight matrix heatmap (if 2D)
        if len(weight.shape) == 2:
            im1 = axes[0,0].imshow(weight, cmap='viridis', aspect='auto')
            axes[0,0].set_title('Weight Matrix Heatmap')
            axes[0,0].set_xlabel('Output Units')
            axes[0,0].set_ylabel('Input Units')
            plt.colorbar(im1, ax=axes[0,0])
        else:
            # For higher dimensional weights, flatten and reshape
            weight_flat = weight.flatten()
            weight_2d = weight_flat.reshape(-1, min(100, len(weight_flat)))
            im1 = axes[0,0].imshow(weight_2d, cmap='viridis', aspect='auto')
            axes[0,0].set_title('Weight Matrix Heatmap (Reshaped)')
            plt.colorbar(im1, ax=axes[0,0])
        
        # 2. Weight distribution histogram
        weight_flat = weight.flatten()
        axes[0,1].hist(weight_flat, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0,1].set_title('Weight Distribution')
        axes[0,1].set_xlabel('Weight Value')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].axvline(weight_flat.mean(), color='red', linestyle='--', label=f'Mean: {weight_flat.mean():.4f}')
        axes[0,1].legend()
        
        # 3. Row sums
        if len(weight.shape) >= 2:
            row_sums = weight.sum(axis=tuple(range(1, len(weight.shape))))
            axes[1,0].plot(row_sums, 'b-', linewidth=2)
            axes[1,0].set_title('Row Sums')
            axes[1,0].set_xlabel('Row Index')
            axes[1,0].set_ylabel('Sum')
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Column sums
        if len(weight.shape) >= 2:
            col_sums = weight.sum(axis=tuple(range(len(weight.shape)-1)))
            axes[1,1].plot(col_sums, 'r-', linewidth=2)
            axes[1,1].set_title('Column Sums')
            axes[1,1].set_xlabel('Column Index')
            axes[1,1].set_ylabel('Sum')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'weight_{i}_visualization.png', dpi=300, bbox_inches='tight')
        print(f"  Saved visualization: weight_{i}_visualization.png")
        plt.close()
    
    print(f"\nAll visualizations saved!")

if __name__ == "__main__":
    visualize_weights()
