# Stage 9.1: Results Integrity and RL Metric Audit Report

**Project**: Quantum Federated Reinforcement Learning (QFRL) for Energy-Efficient Vehicular Edge Computing  
**Audit Target**: Stage 9 Controlled Classical vs. Quantum Benchmark  
**Date**: 2026-10-12  
**Status**: COMPLETE (PASS)  

---

## Executive Summary

This audit rigorously verifies the results, data pipeline, metric calculations, and numerical stability of the Stage 9 controlled experiment comparing the Classical MLP DDPG actor against the hybrid Quantum-Classical DDPG actor. All 18 model checkpoints, raw and canonical metric logs, and mathematical derivations were analyzed.

### Key Audit Findings:
1. **Duplicate Row Cause Identified**: The original 135-row comparison CSV contained two 45-row runs of the Classical model and one 45-row run of the Quantum model. This was caused by the CSV file being opened in append mode (`'a'`) in `experiments/compare_quantum_classical.py` after an initial execution was interrupted at the Quantum stage by a keyword argument mismatch (`noise_scale`). Run 1 Classical and Run 2 Classical are mathematically identical deterministic evaluations (max difference in rewards, energy, and losses is exactly $0.00000000$).
2. **Fixed Completed Bits Validated**: Completed task bits ($1.572864 \times 10^9\text{ bits}$) and task completion ratio ($1.0$) are constant by design due to the physical bit conservation equation ($\text{local\_bits} + \text{offload\_bits} \equiv \text{input\_bits}$) following the Stage 6A removal of artificial timeout drops. The RL policy optimizes the denominator (**Total Energy**), thereby maximizing Energy Efficiency.
3. **Energy Efficiency Formula Verified**: The reported Energy Efficiency ($\text{Mb/J}$) reproduces the exact underlying ratio:
   $$\eta_{\text{EE}} = \frac{\text{Completed Bits}}{\max(E_{\text{total}}, 10^{-6})} \times 10^{-6}$$
   with $< 10^{-6}\text{ Mb/J}$ numerical residual across all rows.
4. **DDPG Numerical Scale Explained**: Critic MSE losses on the order of $10^{10}\text{--}10^{11}$ are mathematically normal and expected given unscaled step rewards ($R \approx 5.5 \times 10^5\text{ bits/J}$) and cumulative returns ($Q^* \approx 1.1 \times 10^7$). Checkpoint parameter audit confirms 100% finite weights, zero NaN/Inf values, and stable parameter norms.
5. **Statistical Rigor Enforced**: Inter-seed variance (across $N=3$ independent random seeds) has been rigorously separated from intra-run learning progression (across 15 training episodes).
6. **Data Artifacts Canonicalized**: The original raw 135-row file is preserved as raw evidence, and a clean 90-row canonical dataset with unique `run_id` and execution tracking has been generated and validated.

---

## 1. Investigation of Duplicate Rows

### 1.1 Root Cause Analysis
In `experiments/compare_quantum_classical.py`, lines 261–265 govern CSV file handling:
```python
file_exists = os.path.exists(csv_path)
csv_file = open(csv_path, 'a' if file_exists else 'w', newline='')
writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
if not file_exists:
    writer.writeheader()
```
The file was opened in **append mode (`'a'`)** if the file already existed.

During the execution on Kaggle:
- **Run 1**: The script executed the Classical model across all 3 seeds (42, 123, 2026; 15 episodes each), writing **45 rows** (rows 2–46). When the Quantum model began execution on Seed 42 Episode 1, a `TypeError` was raised because `DDPGAgent.select_action()` did not accept the `noise_scale` keyword argument. The script crashed, but the 45 classical rows remained on disk.
- **Run 2**: After patching `select_action()` to support `noise_scale`, the script was rerun from the beginning. Because `results/quantum_classical_comparison.csv` already existed from Run 1, the file was opened in append mode. Run 2 executed the Classical model (appending rows 47–91, **45 rows**) and then immediately executed the Quantum model (appending rows 92–136, **45 rows**).
- **Total Rows**: $45 + 45 + 45 = 135\text{ rows}$.

### 1.2 Mathematical Proof of Determinism
Comparing Classical Run 1 (rows 2–46) against Classical Run 2 (rows 47–91) demonstrates that both runs represent bit-for-bit identical deterministic evaluations:

| Metric | Max Absolute Difference ($\text{Run 1} - \text{Run 2}$) | Status |
|---|---|---|
| `episode_reward` | **$0.00000000$** | Bit-exact |
| `energy_efficiency_mbits_per_joule` | **$0.00000000$** | Bit-exact |
| `total_energy` | **$0.00000000$** | Bit-exact |
| `actor_loss` | **$0.00000000$** | Bit-exact |
| `critic_loss` | **$0.00000000$** | Bit-exact |
| `mean_latency` | **$0.00000000$** | Bit-exact |
| `duration_seconds` | $0.030\text{ s}$ | Minor OS timer variation |
| `inference_time_ms` | $0.032\text{ ms}$ | Minor OS timer variation |

### 1.3 Resolution & Data Artifact Strategy
To maintain absolute data integrity:
1. `results/quantum_classical_comparison_raw.csv`: Retains the complete, unedited 135-row file as historical evidence.
2. `results/quantum_classical_comparison_canonical.csv`: Contains the 90 matched, concurrent episodes (45 Classical + 45 Quantum from Run 2) with explicit unique identifiers (`run_id`, `batch_execution_id`).
3. `results/quantum_classical_comparison_all_runs_annotated.csv`: Contains all 135 rows with explicit run annotations (`classical_seed42_run1`, `classical_seed42_run2`, `quantum_seed42_run2`).

---

## 2. Audit of Fixed Metrics (Completed Bits & Task Completion Ratio)

### 2.1 Trace from Action to Task Completion
In `environment/environment.py` (lines 174–237):
1. **Action Input**: The policy outputs continuous actions $\in [-1, 1]^5$, mapped to physical parameters including `offload_ratio` $\in [0, 1]$.
2. **Bit Partitioning**:
   $$\text{local\_bits} = \text{task.input\_bits} \times (1 - \text{offload\_ratio})$$
   $$\text{offload\_bits} = \text{task.input\_bits} \times \text{offload\_ratio}$$
3. **Execution Accounting**:
   $$\text{completed\_bits} = \text{local\_bits} + \text{offload\_bits} = \text{task.input\_bits} \times (1 - \text{ratio} + \text{ratio}) \equiv \text{task.input\_bits}$$
4. **Step-Level Total**:
   $$\text{total\_completed\_bits} = \sum_{i=1}^{30} \text{task}_i.\text{input\_bits} = 30 \times 1,048,576\text{ bits} = 31,457,280\text{ bits}$$
5. **Episode-Level Total (50 Steps)**:
   $$\text{Episode Completed Bits} = 50 \times 31,457,280 = 1,572,864,000\text{ bits} \equiv 1.572864 \times 10^9\text{ bits}$$
6. **Task Completion Ratio**:
   $$\text{task\_completion\_ratio} = \frac{\text{total\_completed\_bits}}{\sum \text{input\_bits}} \equiv 1.0$$

### 2.2 Why There Are No Task Drops
In the Stage 6A revision, the artificial 10-second hard timeout drop was removed because it was dropping valid edge computation without modeling queuing dynamics. Because no secondary queue overflow or deadline drop mechanism was introduced, every generated task bit is processed.

### 2.3 Policy Impact on Objective
Even though `completed_bits` is constant, the policy **strongly controls energy consumption**:
- Local computation energy: $E_{\text{local}} = \kappa f_v^2 C_{\text{local}}$
- Transmission energy: $E_{\text{tx}} = P_{\text{tx}} T_{\text{tx}}$ (governed by IRS phase shifts, transmit power, UAV distance)
- UAV edge computing energy: $E_{\text{uav}} = \kappa f_{\text{uav}}^2 C_{\text{offload}}$

Total energy varies from **$2,741.86\text{ J}$** (optimal policy) to **$22,253.39\text{ J}$** (suboptimal policy). Hence, minimizing total energy directly maximizes Energy Efficiency ($\text{EE} = \frac{1.572864 \times 10^9}{E_{\text{total}}}$).

---

## 3. Audit of Energy Efficiency Metrics & Physical Units

### 3.1 Unit Consistency
- `completed_bits`: Measured in raw bits ($1,572,864,000\text{ bits}$).
- `total_energy`: Measured in Joules ($2,741.86\text{--}22,253.39\text{ J}$).
- `energy_efficiency_mbits_per_joule`:
  $$\eta_{\text{EE}} = \frac{\text{Bits}}{\text{Joules}} \times 10^{-6} = \frac{\text{Mbits}}{\text{Joule}}$$

### 3.2 Exact Numerical Reproduction Test
Evaluating all rows in the dataset confirms bit-exact agreement:
$$\left|\eta_{\text{EE, reported}} - \frac{\text{completed\_bits}}{\text{total\_energy} \times 10^6}\right| < 10^{-6}\text{ Mb/J}$$

Example from Seed 42 Episode 10:
$$\eta_{\text{EE}} = \frac{1,572,864,000}{2741.864 \times 10^6} = 0.57364769... \approx 0.573648\text{ Mb/J} \quad (\text{Matches CSV exactly})$$

### 3.3 Step Reward vs. Episode Energy Efficiency
- In `rl/reward.py`: Step reward is instantaneous energy efficiency:
  $$R_t = \frac{\text{completed\_bits}_t}{\max(E_{\text{total}, t}, 10^{-6})} \approx \frac{31.45 \times 10^6\text{ bits}}{55\text{ J}} \approx 571,000\text{ bits/Joule}$$
- In `compare_quantum_classical.py`:
  - `episode_reward` $= \sum_{t=1}^{50} R_t \approx 50 \times 571,000 \approx 28.5 \times 10^6\text{ bits/Joule}$
  - `mean_step_reward` $= \frac{\text{episode\_reward}}{50} \approx 571,000\text{ bits/Joule}$
  - `energy_efficiency_mbits_per_joule` $= \frac{\sum \text{bits}}{\sum \text{energy}} \times 10^{-6} \approx 0.571\text{ Mb/J}$

All physical quantities and conversions are verified and consistent.

---

## 4. Audit of DDPG Numerical Scales & Stability

### 4.1 Explanation of Large Critic Losses
Observed critic losses range between **$1.8 \times 10^{10}$ and $2.4 \times 10^{11}$**.
- **Return Scale**: In DDPG, the Bellman target is $y_t = R_t + \gamma Q(s_{t+1}, \mu(s_{t+1}))$. With discount factor $\gamma = 0.95$ and unscaled rewards $R_t \approx 5.5 \times 10^5$, expected discounted return is:
  $$Q^* \approx \frac{R}{1 - \gamma} = \frac{5.5 \times 10^5}{0.05} \approx 1.1 \times 10^7$$
- **MSE Loss Scale**: The critic MSE loss is:
  $$\mathcal{L}_{\text{critic}} = \mathbb{E}\left[(Q(s, a) - y)^2\right]$$
  A root-mean-square error (RMSE) of $1.34 \times 10^5$ yields:
  $$\mathcal{L}_{\text{critic}} = (1.34 \times 10^5)^2 \approx 1.8 \times 10^{10}$$
  An estimation error of $1.34 \times 10^5$ represents only **$\approx 1.2\%$ relative error** on a return of $1.1 \times 10^7$.
- Thus, the magnitude of the critic loss is mathematically expected for raw, unscaled physical rewards and does not indicate numerical divergence.

### 4.2 Actor Loss Trajectory
The actor loss is:
$$\mathcal{L}_{\text{actor}} = -\mathbb{E}\left[Q(s, \mu(s))\right]$$
As training progresses, the critic learns that policy actions yield high positive returns ($Q \approx 10^5\text{--}10^6$), driving $\mathcal{L}_{\text{actor}}$ from $-46.4$ (Episode 1) down to $-922,975.5$ (Episode 15). This monotonic decrease is the standard behavior of DDPG policy gradient updates maximizing expected Q-values.

### 4.3 Checkpoint Weight & Norm Verification
All 18 Stage 9 PyTorch checkpoints were audited for numerical integrity:

| Checkpoint Group | Weights Finite | Actor Norm ($\|\theta_A\|_F$) | Critic Norm ($\|\theta_C\|_F$) | Numerical Status |
|---|---|---|---|---|
| Classical (Seed 42) | **100% True** | $6.70 \to 7.84$ | $6.63 \to 170.99$ | Completely Stable |
| Classical (Seed 123) | **100% True** | $7.65 \to 7.65$ | $64.62 \to 163.95$ | Completely Stable |
| Classical (Seed 2026) | **100% True** | $7.54 \to 7.54$ | $64.16 \to 168.64$ | Completely Stable |
| Quantum (Seed 42) | **100% True** | $23.42 \to 23.74$ | $74.65 \to 183.82$ | Completely Stable |
| Quantum (Seed 123) | **100% True** | $21.51 \to 21.87$ | $55.43 \to 154.28$ | Completely Stable |
| Quantum (Seed 2026) | **100% True** | $21.27 \to 21.51$ | $64.95 \to 108.94$ | Completely Stable |

- Zero NaN or Inf values occurred across all parameters and optimizers.
- Parameter norms remained bounded and well-conditioned throughout all 45 episodes per model.

---

## 5. Audit of Comparison Fairness

| Condition | Classical DDPG Baseline | Quantum QFRL Baseline | Fairness Verdict |
|---|---|---|---|
| **Environment Dynamics** | `configs/paper_config.yaml` (30 veh, 1 UAV, 1 IRS) | `configs/paper_config.yaml` (30 veh, 1 UAV, 1 IRS) | **Identical** |
| **Seeds Evaluated** | `[42, 123, 2026]` | `[42, 123, 2026]` | **Identical** |
| **Episode Budget** | Exactly 15 episodes per seed (45 total) | Exactly 15 episodes per seed (45 total) | **Identical** |
| **Steps per Episode** | 50 steps | 50 steps | **Identical** |
| **Critic Architecture** | MLP `(155 -> 64 -> 64 -> 1)` (14,209 params) | MLP `(155 -> 64 -> 64 -> 1)` (14,209 params) | **Identical** |
| **Replay Buffer & Batch** | Capacity 6,000, batch size 32, warmup 32 | Capacity 6,000, batch size 32, warmup 32 | **Identical** |
| **Learning Rates** | Actor: $0.001$, Critic: $0.0025$ (Adam) | Actor: $0.001$, Critic: $0.0025$ (Adam) | **Identical** |
| **Polyak Target Update** | $\tau = 0.005$, $\gamma = 0.95$ | $\tau = 0.005$, $\gamma = 0.95$ | **Identical** |
| **Exploration Noise** | Constant Gaussian ($\sigma = 0.1$) | Constant Gaussian ($\sigma = 0.1$) | **Identical** |
| **Hardware Platform** | Kaggle Tesla T4 GPU (CUDA) | Kaggle Tesla T4 GPU (CUDA, TorchQuantum 0.3.0) | **Identical** |
| **Inference Step Latency** | **$0.352\text{ ms}$** | **$69.56\text{ ms}$** (~197x slower) | Measured physical reality |

The comparison was conducted under rigorously matched, controlled conditions.

---

## 6. Recomputed Results (Separating Seed Variance from Episode Learning)

Following standard RL reporting guidelines, statistics are reported separately for **final converged policy across seeds**, **whole-training across-seed mean**, and **pooled episode variance**:

### 6.1 Final Policy Performance (Episode 15 across the 3 Seeds)
*Evaluates the asymptotic quality of the learned policy after 15 training episodes.*

| Metric | Classical Baseline (Seed 42, 123, 2026) | Classical Across-Seed Mean $\pm$ Std | Quantum Baseline (Seed 42, 123, 2026) | Quantum Across-Seed Mean $\pm$ Std |
|---|---|---|---|---|
| **Energy Efficiency ($\text{Mb/J}$)** | $[0.5250, 0.5465, 0.5315]$ | $\mathbf{0.5343 \pm 0.0110}$ | $[0.5248, 0.5460, 0.0709]$ | $0.3806 \pm 0.2684$ |
| **Total Energy ($\text{J}$)** | $[2995.86, 2878.13, 2959.22]$ | $\mathbf{2944.40 \pm 60.25}$ | $[2996.84, 2880.66, 22192.18]$ | $9356.56 \pm 11116.12$ |
| **Episode Reward ($\times 10^6$)** | $[27.18, 28.03, 27.44]$ | $\mathbf{27.55 \pm 0.44}$ | $[27.17, 28.02, 3.55]$ | $19.58 \pm 13.89$ |
| **Mean Latency ($\text{s}$)** | $[0.2559, 0.2574, 0.2564]$ | $\mathbf{0.2566 \pm 0.0008}$ | $[0.2559, 0.2574, 0.0854]$ | $0.1996 \pm 0.0989$ |
| **Inference Time ($\text{ms}$)** | $[0.363, 0.355, 0.349]$ | $\mathbf{0.3557 \pm 0.0070}$ | $[70.004, 68.803, 70.527]$ | $69.7780 \pm 0.8839$ |

### 6.2 Whole-Training Performance (Mean of 15 Episodes per Seed, Evaluated Across Seeds)
*Evaluates overall learning progress across the training lifetime.*

| Metric | Classical Baseline Across Seeds | Quantum Baseline Across Seeds |
|---|---|---|
| **Energy Efficiency ($\text{Mb/J}$)** | $\mathbf{0.5231 \pm 0.0050}$ | $0.3478 \pm 0.2378$ |
| **Total Energy ($\text{J}$)** | $\mathbf{3102.60 \pm 54.02}$ | $9546.86 \pm 10284.77$ |
| **Episode Reward ($\times 10^6$)** | $\mathbf{27.28 \pm 0.21}$ | $17.95 \pm 12.37$ |
| **Duration per Episode ($\text{s}$)** | $\mathbf{0.3064 \pm 0.0021}$ | $14.3793 \pm 0.0336$ |

### 6.3 Pooled Episode Variance (Across All 45 Canonical Episodes)
*Conflates intra-run learning progression (episodes 1 &rarr; 15) with seed variability.*
- **Classical Pooled EE**: $0.5231 \pm 0.0697\text{ Mb/J}$
- **Quantum Pooled EE**: $0.3478 \pm 0.2157\text{ Mb/J}$

---

## 7. Conclusions & Final Scientific Answer

1. **Does the Quantum Actor Offer an Empirical Advantage?**
   - **NO.** When evaluated under controlled conditions, the quantum actor achieves virtually identical peak energy efficiency ($0.573\text{ Mb/J}$ vs. $0.574\text{ Mb/J}$) and identical converged policies on Seeds 42 and 123, but fails to escape suboptimal exploration on Seed 2026.
   - The classical MLP actor converges faster (Episode 2 vs. Episode 5–7), exhibits 100% seed stability, and runs **~197x faster per action** on Tesla T4 GPU.
2. **What Is the Distinguishing Empirical Property of the Quantum Actor?**
   - **Parameter Compactness**: The quantum actor achieves comparable peak energy efficiency with only **773 parameters** vs. **14,149 parameters** for the classical MLP (a **94.5% parameter reduction**).
   - This compact representation is highly relevant for federated learning bandwidth constraints, which will be evaluated in Stage 10.

---

## 8. Artifact Register

- **Stage 9.1 Audit Report**: [`STAGE9_1_RESULTS_AUDIT.md`](file:///d:/Study%20Material/Research/QRFL/QFRL_Highway/STAGE9_1_RESULTS_AUDIT.md)
- **Stage 9.2 Reward & Dynamics Audit Report**: [`STAGE9_2_REWARD_OBJECTIVE_AUDIT.md`](file:///d:/Study%20Material/Research/QRFL/QFRL_Highway/STAGE9_2_REWARD_OBJECTIVE_AUDIT.md)
- **Raw Comparison Data (135 rows)**: [`results/quantum_classical_comparison_raw.csv`](file:///d:/Study%20Material/Research/QRFL/QFRL_Highway/results/quantum_classical_comparison_raw.csv)
- **Canonical Comparison Data (90 rows)**: [`results/quantum_classical_comparison_canonical.csv`](file:///d:/Study%20Material/Research/QRFL/QFRL_Highway/results/quantum_classical_comparison_canonical.csv)
- **Annotated Multi-Run Data (135 rows)**: [`results/quantum_classical_comparison_all_runs_annotated.csv`](file:///d:/Study%20Material/Research/QRFL/QFRL_Highway/results/quantum_classical_comparison_all_runs_annotated.csv)
- **PyTorch Checkpoints**: All 18 `.pt` models preserved in [`results/checkpoints/`](file:///d:/Study%20Material/Research/QRFL/QFRL_Highway/results/checkpoints/)
- **Audit Test Suite**: [`tests/test_stage9_audit.py`](file:///d:/Study%20Material/Research/QRFL/QFRL_Highway/tests/test_stage9_audit.py) & [`tests/test_reward_objective.py`](file:///d:/Study%20Material/Research/QRFL/QFRL_Highway/tests/test_reward_objective.py) (All 5 audit tests pass)

---

## 9. Stage 9.2 Linked Corrections & Qualification

*Detailed derivations and checkpoint delta measurements are documented in [`STAGE9_2_REWARD_OBJECTIVE_AUDIT.md`](file:///d:/Study%20Material/Research/QRFL/QFRL_Highway/STAGE9_2_REWARD_OBJECTIVE_AUDIT.md).*

1. **Reward vs. Aggregate EE Equivalence**:
   Cumulative step reward optimizes the Harmonic Mean of step energies, while aggregate EE optimizes the Arithmetic Mean. By Jensen's inequality, $\frac{J_{\text{RL}}}{T \times 10^6} \ge \eta_{\text{EE}}$. Exactly 33 rank inversions (3.33%) exist in the Classical data, showing they are not strictly rank-preserving.
2. **Convergence Status Qualification**:
   Episode 15 is strictly the final episode of a short 15-episode pilot training horizon. The Classical actor froze by Episode 5 due to Tanh activation saturation at boundaries ($\|\Delta \theta_A(10 \to 15)\| \approx 0.0000$), whereas the Quantum actor maintained active plasticity ($\|\Delta \theta_A(10 \to 15)\| \approx 2.2\%\text{--}3.6\%$).
3. **Conditioning on $N=3$**:
   All comparative claims are strictly conditioned on the sample of three seeds (`[42, 123, 2026]`).
4. **Parameter Footprint vs. Federated Communication**:
   The 94.5% parameter reduction is a model architectural attribute; empirical proof of communication savings requires federated aggregation trials under bandwidth constraints (Stage 10).
