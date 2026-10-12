#!/usr/bin/env python3
"""
STAGE 9: SEPARATE EXPLORATION SCHEDULE EXPERIMENT

Investigates whether scheduled exploration noise decay mitigates action saturation
(e.g., observed in Seed 2026 during Stage 8) compared to constant Gaussian noise.

Compares:
1. Schedule A: Constant Noise (sigma = 0.1, Stage 8 default)
2. Schedule B: Exponential Decaying Noise (sigma_t = max(0.01, 0.1 * 0.95^(ep-1)))

Keeps all other environment and model hyperparameters strictly fixed.
"""

import os
import sys
import copy
import yaml
import argparse
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_root)

from experiments.compare_quantum_classical import run_controlled_comparison

def main():
    parser = argparse.ArgumentParser(description="Exploration Schedule Comparison")
    parser.add_argument("--config", type=str, default="configs/paper_config.yaml")
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--models", nargs="+", type=str, default=["classical", "quantum"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow_mock", action="store_true")
    args = parser.parse_args()
    
    config_path = os.path.join(repo_root, args.config)
    
    print("=" * 80)
    print("EXPLORATION SCHEDULE EXPERIMENT: CONSTANT VS. DECAYING NOISE")
    print("=" * 80)
    
    # 1. Run Schedule A: Constant Noise
    print("\n[PART 1] Running Schedule A: Constant Exploration Noise (sigma = 0.1)...")
    run_controlled_comparison(
        config_path=config_path,
        models=args.models,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        seeds=args.seeds,
        device=args.device,
        output_dir="results/exploration_constant",
        allow_mock=args.allow_mock,
        noise_schedule="constant"
    )
    
    # 2. Run Schedule B: Decaying Noise
    print("\n[PART 2] Running Schedule B: Decaying Exploration Noise (sigma_0 = 0.1, decay = 0.95)...")
    run_controlled_comparison(
        config_path=config_path,
        models=args.models,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        seeds=args.seeds,
        device=args.device,
        output_dir="results/exploration_decay",
        allow_mock=args.allow_mock,
        noise_schedule="decay",
        decay_rate=0.95
    )
    
    print("\n" + "=" * 80)
    print("EXPLORATION SCHEDULE EXPERIMENT COMPLETE.")
    print("=" * 80)

if __name__ == "__main__":
    main()
