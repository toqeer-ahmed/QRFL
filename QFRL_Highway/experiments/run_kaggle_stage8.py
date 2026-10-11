import os
import sys
import time

# Ensure repository root is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# If running on Kaggle, ensure working directory is repo root
if os.path.exists('/kaggle/working/QRFL/QFRL_Highway'):
    os.chdir('/kaggle/working/QRFL/QFRL_Highway')

from experiments.train_baseline_qfrl import (
    verify_backend,
    check_device,
    run_pilot_smoke_test,
    run_baseline_training
)
import yaml
import numpy as np
import pandas as pd

def main():
    print("==================================================")
    print("STAGE 8: BASELINE QFRL TRAINING AND EVALUATION")
    print("==================================================")
    
    # 1. Backend Verification (strictly real, fails loudly on mock)
    tq, backend_name, is_mock = verify_backend(allow_mock=False)
    active_device = check_device('cuda')
    
    config_path = "configs/paper_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"Active Device       : {active_device}")
    print(f"Backend Name        : {backend_name}")
    print(f"Paper Config Loaded : {config_path}")
    
    # 2. Pilot Smoke Test (Seed 42)
    print("\n--- STEP 1: RUNNING PILOT SMOKE TEST ---")
    pilot_ok = run_pilot_smoke_test(
        env_config=config,
        device=active_device,
        seed=42,
        is_mock=False
    )
    if not pilot_ok:
        print("PILOT FAILED! Stopping.")
        sys.exit(1)
    print("[PASS] Pilot smoke test succeeded on real TorchQuantum backend.")
    
    # 3. Controlled Multi-Seed Baseline Training
    # Seeds: 42, 123, 2026 | 15 episodes per seed | 50 steps per episode
    seeds = [42, 123, 2026]
    episodes = 15
    steps_per_episode = 50
    output_dir = "results"
    
    print(f"\n--- STEP 2: RUNNING MULTI-SEED BASELINE TRAINING ---")
    print(f"Seeds: {seeds} | Episodes per seed: {episodes} | Steps per episode: {steps_per_episode}")
    
    start_time = time.time()
    all_metrics, total_duration = run_baseline_training(
        config_path=config_path,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        seeds=seeds,
        device=active_device,
        output_dir=output_dir,
        allow_mock=False,
        checkpoint_freq=5,
        config_id="paper_baseline_multiseed",
        backend_name=backend_name
    )
    
    # 4. Statistical Aggregation
    csv_path = os.path.join(output_dir, "baseline_qfrl_metrics.csv")
    df = pd.read_csv(csv_path)
    
    print("\n==================================================")
    print("STAGE 8 BASELINE EVALUATION SUMMARY")
    print("==================================================")
    print(f"Total Runs / Seeds   : {len(seeds)} ({seeds})")
    print(f"Total Episodes       : {len(df)}")
    print(f"Total Runtime        : {total_duration:.2f} seconds ({total_duration/60:.2f} min)")
    print(f"Average Time/Episode : {total_duration/len(df):.2f} seconds")
    
    print("\n--- METRICS ACROSS ALL EPISODES (MEAN +/- STD) ---")
    print(f"Episode Reward       : {df['episode_reward'].mean():.2f} +/- {df['episode_reward'].std():.2f}")
    print(f"Energy Efficiency    : {df['energy_efficiency_mbits_per_joule'].mean():.4f} +/- {df['energy_efficiency_mbits_per_joule'].std():.4f} Mbits/J")
    print(f"Total Energy         : {df['total_energy'].mean():.2f} +/- {df['total_energy'].std():.2f} J")
    print(f"Completed Bits       : {df['completed_bits'].mean():.2e} +/- {df['completed_bits'].std():.2e} bits")
    print(f"Mean Latency         : {df['mean_latency'].mean():.4f} +/- {df['mean_latency'].std():.4f} s")
    
    # Per-seed analysis
    print("\n--- PER-SEED PERFORMANCE ---")
    for s in seeds:
        seed_df = df[df['seed'] == s]
        initial_rew = seed_df.iloc[0]['episode_reward']
        final_rew = seed_df.iloc[-1]['episode_reward']
        mean_rew = seed_df['episode_reward'].mean()
        mean_ee = seed_df['energy_efficiency_mbits_per_joule'].mean()
        print(f"Seed {s:4d} | Init Rew: {initial_rew:9.2f} | Final Rew: {final_rew:9.2f} | Mean Rew: {mean_rew:9.2f} | Mean EE: {mean_ee:.4f} Mb/J")
        
    print("\nArtifacts Saved:")
    print(f"1. Metrics CSV       : {csv_path}")
    print(f"2. Learning Curves   : {os.path.join(output_dir, 'baseline_qfrl_learning_curves.png')}")
    print(f"3. Config Snapshot   : {os.path.join(output_dir, 'config_snapshot.yaml')}")
    print(f"4. Checkpoints Dir   : {os.path.join(output_dir, 'checkpoints')}")
    print("==================================================")
    print("STAGE 8 BASELINE QFRL RUN COMPLETE.")

if __name__ == "__main__":
    main()
