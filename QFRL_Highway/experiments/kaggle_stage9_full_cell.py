# ==============================================================================
# STAGE 9: CONTROLLED QUANTUM VS. CLASSICAL DDPG BENCHMARK (ALL-IN-ONE CELL)
# ==============================================================================
# Run this entire block in a single cell of your Kaggle Notebook with GPU enabled (Tesla T4).

import os
import sys
import shutil

# Step 0: Set working directory and update/clone repository
os.chdir('/kaggle/working')
repo_dir = '/kaggle/working/QRFL'

if not os.path.exists(repo_dir):
    print("Cloning repository...")
    !git clone https://github.com/toqeer-ahmed/QRFL.git
else:
    print("Repository exists, syncing to clean origin/main...")
    %cd /kaggle/working/QRFL
    !git fetch origin
    !git reset --hard origin/main
    %cd /kaggle/working

qfrl_dir = os.path.join(repo_dir, 'QFRL_Highway')
os.chdir(qfrl_dir)
if qfrl_dir not in sys.path:
    sys.path.insert(0, qfrl_dir)

# Step 1: Ensure required dependencies are installed
!pip install -q --no-deps torchpack torchdiffeq
!pip install -q qiskit-ibm-runtime qiskit-aer loguru multimethod tensorpack toml
!pip install -q --no-deps git+https://github.com/mit-han-lab/torchquantum.git

# Step 2: Run Stage 9 Controlled Benchmark
# Executes:
# 1. Pilot smoke test for Classical and Quantum agents
# 2. Matched 15-episode evaluation across seeds [42, 123, 2026] for BOTH models
# 3. Saves results/quantum_classical_comparison.csv & results/quantum_classical_comparison.png
!python -u experiments/compare_quantum_classical.py \
    --models classical quantum \
    --episodes 15 \
    --steps 50 \
    --seeds 42 123 2026 \
    --device cuda \
    --output_dir results

# Step 3: Export Results, Generate 1-Click ZIP Download, and Display Plot
import pandas as pd
from IPython.display import Image, display, FileLink

src_results = '/kaggle/working/QRFL/QFRL_Highway/results'
dst_results = '/kaggle/working/results'
zip_output = '/kaggle/working/qfrl_stage9_results'

if os.path.exists(src_results):
    if os.path.exists(dst_results):
        shutil.rmtree(dst_results)
    shutil.copytree(src_results, dst_results)
    shutil.make_archive(zip_output, 'zip', src_results)
    
    print("\n" + "=" * 80)
    print("STAGE 9 BENCHMARK RUN COMPLETE!")
    print("=" * 80)
    print("⬇️ Click link below to download full Stage 9 results (CSV, plots, checkpoints):")
    display(FileLink("qfrl_stage9_results.zip"))
    
    plot_file = os.path.join(src_results, "quantum_classical_comparison.png")
    if os.path.exists(plot_file):
        print("\n📈 Side-by-Side Comparison Learning Curves:")
        display(Image(filename=plot_file, width=900))
        
    csv_file = os.path.join(src_results, "quantum_classical_comparison.csv")
    if os.path.exists(csv_file):
        print("\n📊 Summary of Comparison Metrics (First 10 rows):")
        df = pd.read_csv(csv_file)
        display(df[['model_type', 'seed', 'episode', 'energy_efficiency_mbits_per_joule', 'total_energy', 'episode_reward']].head(10))
