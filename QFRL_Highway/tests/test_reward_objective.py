import os
import pytest
import pandas as pd
import numpy as np

def test_jensen_inequality_reward_vs_ee():
    """
    Verifies the mathematical inequality between cumulative step reward sum
    and aggregate episode energy efficiency:
    Sum of step rewards = B * sum(1/E_t) = B * T / HM(E_t)
    Aggregate EE = T * B / sum(E_t) = B / AM(E_t)
    By Jensen's inequality: 1/HM(E_t) >= 1/AM(E_t),
    so normalized reward sum must be >= aggregate EE for every episode.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "..", "results", "quantum_classical_comparison_canonical.csv")
    df = pd.read_csv(csv_path)
    
    # Normalized reward in Mb/J:
    # ep_reward is in bits/J. Dividing by 50 (steps) and 1e6 gives the mean step reward in Mb/J.
    mean_step_reward_mb = (df['episode_reward'] / 50.0) / 1e6
    aggregate_ee_mb = df['energy_efficiency_mbits_per_joule']
    
    # Check that mean_step_reward_mb >= aggregate_ee_mb within numerical precision
    diff = mean_step_reward_mb - aggregate_ee_mb
    assert np.all(diff >= -1e-6), "Jensen's inequality violated: mean step reward < aggregate EE"
    
    # Check that diff is strictly positive on episodes with non-constant energy
    assert np.mean(diff) > 0.0, "Expected non-zero Jensen gap across episodes"

def test_pairwise_rank_inversion_presence():
    """
    Demonstrates that maximizing cumulative step reward is not strictly
    equivalent to maximizing aggregate episode EE, as proven by the presence
    of rank inversions in real execution data.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "..", "results", "quantum_classical_comparison_canonical.csv")
    df = pd.read_csv(csv_path)
    
    c_df = df[df['model_type'] == 'classical']
    inversions = 0
    total_pairs = 0
    records = c_df.to_dict('records')
    
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            total_pairs += 1
            r_diff = records[i]['episode_reward'] - records[j]['episode_reward']
            ee_diff = records[i]['energy_efficiency_mbits_per_joule'] - records[j]['energy_efficiency_mbits_per_joule']
            if (r_diff > 0 and ee_diff < 0) or (r_diff < 0 and ee_diff > 0):
                inversions += 1
                
    assert inversions > 0, "Expected rank inversions between step reward and aggregate EE"
    assert inversions == 33, f"Expected exactly 33 rank inversions for Classical run, got {inversions}"
