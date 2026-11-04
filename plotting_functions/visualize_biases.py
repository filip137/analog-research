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

def visualize_biases():
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
    
    # Get bias parameters
    bias_params = [param for param in energy_fn.params() if "Bias" in param.name]
    
    print(f"Found {len(bias_params)} bias vectors")
    
    # Visualize each bias vector
    for i, bias_param in enumerate(bias_params):
        bias = bias_param.state.detach().cpu().numpy()
        print(f"\nBias {i} ({bias_param.name}):")
        print(f"  Shape: {bias.shape}")
        print(f"  Min: {bias.min():.6f}")
        print(f"  Max: {bias.max():.6f}")
        print(f"  Mean: {bias.mean():.6f}")
        print(f"  Std: {bias.std():.6f}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Bias Vector {i} ({bias_param.name})', fontsize=16)
        
        # 1. Bias values plot
        axes[0,0].plot(bias.flatten(), 'b-', linewidth=2, marker='o', markersize=3)
        axes[0,0].set_title('Bias Values')
        axes[0,0].set_xlabel('Unit Index')
        axes[0,0].set_ylabel('Bias Value')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].axhline(0, color='red', linestyle='--', alpha=0.5)
        
        # 2. Bias distribution histogram
        bias_flat = bias.flatten()
        axes[0,1].hist(bias_flat, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0,1].set_title('Bias Distribution')
        axes[0,1].set_xlabel('Bias Value')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].axvline(bias_flat.mean(), color='red', linestyle='--', 
                         label=f'Mean: {bias_flat.mean():.4f}')
        axes[0,1].legend()
        
        # 3. Absolute bias values
        abs_bias = np.abs(bias_flat)
        axes[1,0].plot(abs_bias, 'r-', linewidth=2, marker='s', markersize=3)
        axes[1,0].set_title('Absolute Bias Values')
        axes[1,0].set_xlabel('Unit Index')
        axes[1,0].set_ylabel('|Bias Value|')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Bias statistics
        stats_text = f"""Bias Statistics:
Min: {bias.min():.6f}
Max: {bias.max():.6f}
Mean: {bias.mean():.6f}
Std: {bias.std():.6f}
Median: {np.median(bias_flat):.6f}
Range: {bias.max() - bias.min():.6f}
Non-zero: {np.count_nonzero(bias_flat)}/{len(bias_flat)}"""
        
        axes[1,1].text(0.1, 0.5, stats_text, transform=axes[1,1].transAxes, 
                      fontsize=12, verticalalignment='center',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        axes[1,1].set_title('Bias Statistics')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'bias_{i}_visualization.png', dpi=300, bbox_inches='tight')
        print(f"  Saved visualization: bias_{i}_visualization.png")
        plt.close()
    
    print(f"\nAll bias visualizations saved!")

if __name__ == "__main__":
    visualize_biases()
