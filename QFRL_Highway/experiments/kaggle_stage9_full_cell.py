# ==============================================================================
# STAGE 9: CONTROLLED QUANTUM VS. CLASSICAL DDPG BENCHMARK (KAGGLE RUNNER CELL)
# ==============================================================================
# Run this entire block in a single cell of your Kaggle Notebook with GPU enabled (Tesla T4).

import os
import sys

# Step 0: Set working directory and pull/clone repo
os.chdir('/kaggle/working')
repo_dir = '/kaggle/working/QRFL'

if not os.path.exists(repo_dir):
    print("Cloning repository...")
    !git clone https://github.com/toqeer-ahmed/QRFL.git
else:
    print("Repository exists, updating...")
    %cd /kaggle/working/QRFL
    !git pull
    %cd /kaggle/working

# Ensure QFRL_Highway is in sys.path
qfrl_dir = os.path.join(repo_dir, 'QFRL_Highway')
os.chdir(qfrl_dir)
if qfrl_dir not in sys.path:
    sys.path.insert(0, qfrl_dir)

# Step 1: Ensure required dependencies are installed
!pip install -q qiskit-ibm-runtime qiskit-aer loguru multimethod tensorpack toml
!pip install -q --no-deps git+https://github.com/mit-han-lab/torchquantum.git

# Step 2: Run Stage 9 Controlled Benchmark
# Executes:
# 1. Pilot smoke test for Classical and Quantum agents
# 2. Matched 15-episode comparison across seeds [42, 123, 2026]
# 3. Saves results/quantum_classical_comparison.csv & results/quantum_classical_comparison.png
!python -u experiments/run_kaggle_stage9.py
