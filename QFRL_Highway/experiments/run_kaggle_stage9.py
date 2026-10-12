#!/usr/bin/env python3
"""
STAGE 9: CONTROLLED QUANTUM VS. CLASSICAL DDPG BENCHMARK (KAGGLE RUNNER)

Automated execution script for Kaggle environment with NVIDIA GPU.
Verifies backend, runs pilot smoke tests, executes matched 15-episode comparison,
and evaluates decaying exploration schedule.
"""

import os
import sys
import argparse

# Ensure repo root is on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_root)

from experiments.compare_quantum_classical import (
    run_pilot_smoke_test,
    run_controlled_comparison,
    verify_quantum_backend
)
import yaml
import torch

def main():
    print("=" * 80)
    print("STAGE 9: CONTROLLED QUANTUM VS. CLASSICAL DDPG BENCHMARK")
    print("=" * 80)
    
    config_path = os.path.join(repo_root, "configs", "paper_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Active Device       : {device}")
    
    # 1. Pilot Smoke Test
    print("\n--- STEP 1: RUNNING PILOT SMOKE TEST ---")
    pilot_ok = run_pilot_smoke_test(
        env_config=config,
        device=device,
        allow_mock=False
    )
    if not pilot_ok:
        print("[FAIL] Pilot smoke test failed. Exiting.")
        sys.exit(1)
        
    # 2. Matched Primary Benchmark (Constant Exploration Noise = 0.1)
    print("\n--- STEP 2: RUNNING MATCHED PRIMARY BENCHMARK (CONSTANT NOISE) ---")
    seeds = [42, 123, 2026]
    episodes = 15
    steps = 50
    
    run_controlled_comparison(
        config_path=config_path,
        models=["classical", "quantum"],
        episodes=episodes,
        steps_per_episode=steps,
        seeds=seeds,
        device=device,
        output_dir="results",
        allow_mock=False,
        noise_schedule="constant"
    )
    
    print("\n" + "=" * 80)
    print("STAGE 9 BENCHMARK EXECUTION FINISHED.")
    print("=" * 80)

if __name__ == "__main__":
    main()
