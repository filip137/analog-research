#!/usr/bin/env python3
"""
Script to trace where TensorBoard events files are generated.
"""

import json
import os

def trace_events_path():
    print("=== TENSORBOARD EVENTS FILE PATH TRACING ===\n")
    
    # 1. Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    print("1. Config loaded from config.json")
    print(f"   Base path: {config['paths']['base_path']}")
    
    # 2. Path construction in drn_config.py
    base_path = config["paths"]["base_path"]  # "papers/fast-drn"
    model = "drn-xs"
    algorithm = "EP"
    path = "/".join([base_path, model, algorithm])  # "papers/fast-drn/drn-xs/EP"
    
    print(f"\n2. Path construction in drn_config.py:")
    print(f"   base_path = config['paths']['base_path']  # '{base_path}'")
    print(f"   model = '{model}'")
    print(f"   algorithm = '{algorithm}'")
    print(f"   path = '/'.join([base_path, model, algorithm])  # '{path}'")
    
    # 3. Monitor initialization
    print(f"\n3. Monitor initialization in drn_config.py:")
    print(f"   monitor = Monitor(trainer=trainer, evaluator=evaluator, scheduler=scheduler, save_model=energy_fn.save, path=path)")
    print(f"   # path = '{path}'")
    
    # 4. SummaryWriter creation in Monitor.__init__
    print(f"\n4. SummaryWriter creation in training/lab/monitor.py:")
    print(f"   self._writer = SummaryWriter(self._path)")
    print(f"   # self._path = '{path}'")
    
    # 5. Actual file location
    full_path = os.path.abspath(path)
    print(f"\n5. Actual events file location:")
    print(f"   Full path: {full_path}")
    print(f"   Events files: {full_path}/events.out.tfevents*")
    
    # 6. Check if path exists
    if os.path.exists(full_path):
        events_files = [f for f in os.listdir(full_path) if f.startswith('events.out.tfevents')]
        print(f"\n6. Found {len(events_files)} events files in this directory")
        if events_files:
            print("   Recent files:")
            for f in sorted(events_files)[-3:]:  # Show last 3
                file_path = os.path.join(full_path, f)
                size = os.path.getsize(file_path)
                print(f"     - {f} ({size} bytes)")
    else:
        print(f"\n6. Directory does not exist: {full_path}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Events files are generated in: {full_path}")
    print(f"TensorBoard command: tensorboard --logdir={full_path}")

if __name__ == "__main__":
    trace_events_path()
