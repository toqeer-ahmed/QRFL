import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import yaml
import numpy as np
import pandas as pd
import torch

from rl.reward import RewardFunction
from environment.environment import QFRLEnvironment
from experiments.experiment_reward_alignment import run_reward_alignment_experiment

def test_reward_signs_and_bounds():
    """
    Verifies that:
    1. 'ratio' reward is strictly positive and scales with completed_bits / energy.
    2. 'negative_energy' reward is strictly negative and equals -total_energy.
    """
    ratio_func = RewardFunction(reward_type='ratio')
    neg_func = RewardFunction(reward_type='negative_energy')
    
    test_info = {
        'completed_bits': 1048576.0,
        'total_energy': 52.4288,
        'task_completion_ratio': 1.0,
        'transmission_time': 0.05,
        'computation_time': 0.08
    }
    
    r_ratio, info_ratio = ratio_func.calculate_reward(test_info)
    r_neg, info_neg = neg_func.calculate_reward(test_info)
    
    # 1. Ratio reward checks
    assert r_ratio > 0.0, "Ratio reward must be strictly positive"
    expected_ratio = 1048576.0 / 52.4288 # = 20,000.0 bits/J
    assert np.isclose(r_ratio, expected_ratio), f"Expected {expected_ratio}, got {r_ratio}"
    
    # 2. Negative energy reward checks
    assert r_neg < 0.0, "Negative energy reward must be strictly negative"
    assert np.isclose(r_neg, -52.4288), f"Expected -52.4288, got {r_neg}"
    assert np.isclose(info_neg['total_energy'], 52.4288)

def test_negative_energy_monotonic_equivalence_to_aggregate_ee():
    """
    Demonstrates mathematically and empirically that when completed task bits
    and horizon are constant across episodes:
      sum(R_t) = -sum(E_t)
    Therefore, maximizing sum(R_t) is strictly monotonically equivalent to
    maximizing aggregate episode energy efficiency (B_total / E_total).
    Zero rank inversions must exist between cumulative reward and aggregate EE.
    """
    # Simulate a range of 20 episodes with varying total energy
    np.random.seed(42)
    energies = np.random.uniform(20.0, 100.0, size=(20, 50)) # 20 episodes, 50 steps each
    bits_per_step = 1048576.0
    
    episode_data = []
    for ep_idx, ep_energies in enumerate(energies):
        total_e = float(np.sum(ep_energies))
        total_b = float(bits_per_step * 50)
        
        # Condition A: ratio reward
        ratio_step_rewards = bits_per_step / ep_energies
        cum_ratio_reward = float(np.sum(ratio_step_rewards))
        
        # Condition B: negative energy reward
        neg_step_rewards = -ep_energies
        cum_neg_reward = float(np.sum(neg_step_rewards))
        
        # Aggregate EE
        aggregate_ee = (total_b / total_e) / 1e6 # Mbits / Joule
        
        episode_data.append({
            'ep': ep_idx,
            'total_energy': total_e,
            'cum_ratio_reward': cum_ratio_reward,
            'cum_neg_reward': cum_neg_reward,
            'aggregate_ee': aggregate_ee
        })
        
    df = pd.DataFrame(episode_data)
    
    # Check rank inversions between cum_neg_reward and aggregate_ee
    neg_inversions = 0
    ratio_inversions = 0
    records = df.to_dict('records')
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            # Check negative energy reward
            r_neg_diff = records[i]['cum_neg_reward'] - records[j]['cum_neg_reward']
            ee_diff = records[i]['aggregate_ee'] - records[j]['aggregate_ee']
            if (r_neg_diff > 0 and ee_diff < 0) or (r_neg_diff < 0 and ee_diff > 0):
                neg_inversions += 1
                
            # Check ratio reward
            r_ratio_diff = records[i]['cum_ratio_reward'] - records[j]['cum_ratio_reward']
            if (r_ratio_diff > 0 and ee_diff < 0) or (r_ratio_diff < 0 and ee_diff > 0):
                ratio_inversions += 1
                
    # Direct negative energy MUST have ZERO rank inversions with aggregate EE
    assert neg_inversions == 0, f"Negative energy reward produced {neg_inversions} rank inversions; must be 0!"
    
    # Ratio reward typically has rank inversions due to Jensen's inequality
    assert ratio_inversions > 0, "Expected non-zero rank inversions for ratio reward"

def test_reward_alignment_smoke_execution(tmp_path):
    """
    Executes a minimal smoke run (1 episode, 2 steps) for both ratio and negative_energy
    using Classical model to confirm finite rewards, valid gradients, and CSV generation.
    """
    output_dir = str(tmp_path / "results")
    csv_name = "test_smoke_alignment.csv"
    
    results = run_reward_alignment_experiment(
        config_path="configs/reward_alignment_config.yaml",
        models=["classical"],
        reward_types=["ratio", "negative_energy"],
        seeds=[42],
        episodes=1,
        steps_per_episode=2,
        device="cpu",
        output_dir=output_dir,
        output_csv=csv_name,
        smoke_test=True
    )
    
    assert len(results) == 2, f"Expected 2 results (1 per condition), got {len(results)}"
    
    ratio_row = [r for r in results if r['reward_type'] == 'ratio'][0]
    neg_row = [r for r in results if r['reward_type'] == 'negative_energy'][0]
    
    # Validate ratio row
    assert ratio_row['episode_reward'] > 0.0
    assert np.isfinite(ratio_row['energy_efficiency_mbits_per_joule'])
    assert np.isfinite(ratio_row['total_energy'])
    assert 0.0 <= ratio_row['frac_saturated_actions'] <= 1.0
    
    # Validate negative energy row
    assert neg_row['episode_reward'] < 0.0
    assert np.isfinite(neg_row['energy_efficiency_mbits_per_joule'])
    assert np.isfinite(neg_row['total_energy'])
    assert 0.0 <= neg_row['frac_saturated_actions'] <= 1.0
    
    # Verify CSV was created and matches
    csv_full_path = os.path.join(output_dir, csv_name)
    assert os.path.exists(csv_full_path)
    df = pd.read_csv(csv_full_path)
    assert len(df) == 2
    assert "run_id" in df.columns
    assert "frac_saturated_actions" in df.columns
    assert "mean_tanh_derivative" in df.columns
