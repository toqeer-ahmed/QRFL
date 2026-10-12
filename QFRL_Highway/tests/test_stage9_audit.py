import os
import pytest
import pandas as pd
import numpy as np

def test_canonical_csv_structure_and_row_count():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "results", "quantum_classical_comparison_canonical.csv")
    assert os.path.exists(csv_path), "Canonical comparison CSV missing"
    
    df = pd.read_csv(csv_path)
    assert len(df) == 90, f"Expected 90 canonical rows, got {len(df)}"
    
    assert set(df['model_type'].unique()) == {'classical', 'quantum'}
    assert set(df['seed'].unique()) == {42, 123, 2026}
    
    # 45 rows per model
    assert len(df[df['model_type'] == 'classical']) == 45
    assert len(df[df['model_type'] == 'quantum']) == 45
    
    # Check run_id uniqueness per model-seed-episode
    unique_keys = df['model_type'] + '_' + df['seed'].astype(str) + '_ep' + df['episode'].astype(str)
    assert len(unique_keys.unique()) == 90, "Duplicate episode keys found in canonical CSV"

def test_energy_efficiency_formula_reproducibility():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "results", "quantum_classical_comparison_canonical.csv")
    df = pd.read_csv(csv_path)
    
    computed_ee = (df['completed_bits'] / df['total_energy']) / 1e6
    reported_ee = df['energy_efficiency_mbits_per_joule']
    
    # Check max difference within rounding error (4-6 decimal places)
    max_diff = np.max(np.abs(computed_ee - reported_ee))
    assert max_diff < 1e-4, f"EE formula mismatch: max diff {max_diff}"

def test_completed_bits_invariance():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "results", "quantum_classical_comparison_canonical.csv")
    df = pd.read_csv(csv_path)
    
    # 30 vehicles * 1,048,576 bits * 50 steps = 1,572,864,000 bits
    expected_bits = 1572864000.0
    assert np.allclose(df['completed_bits'], expected_bits), "Completed bits physically deviate from expected total"
    assert np.allclose(df['task_completion_ratio'], 1.0), "Task completion ratio deviates from 1.0"
