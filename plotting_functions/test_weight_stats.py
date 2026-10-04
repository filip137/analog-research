#!/usr/bin/env python3
import sys
import os
sys.path.append('../../')

from training.lab.statistics import WeightDistributionStat
import torch

# Test the WeightDistributionStat class
def test_weight_distribution_stat():
    print("Testing WeightDistributionStat...")
    
    # Create a mock variable with weight data
    class MockVariable:
        def __init__(self, name, weight_data):
            self.name = name
            self.state = weight_data
    
    # Create test weight data
    weight_data = torch.randn(100, 50)  # 100x50 weight matrix
    mock_var = MockVariable("DenseWeight_0", weight_data)
    
    # Test different statistics
    stat_types = ['mean', 'std', 'min', 'max', 'abs_mean', 'abs_std']
    
    for stat_type in stat_types:
        stat = WeightDistributionStat(mock_var, stat_type)
        value = stat._measure_fn()
        print(f"{stat_type}: {value:.6f}")
    
    print("WeightDistributionStat test completed successfully!")

if __name__ == "__main__":
    test_weight_distribution_stat()
