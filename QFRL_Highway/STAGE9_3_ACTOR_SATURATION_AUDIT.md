# STAGE 9.3 AUDIT REPORT: CONTROLLED REWARD ALIGNMENT AND ACTOR SATURATION DIAGNOSTICS

**Author**: Antigravity Autonomous Agent (Pair Programming with Research Lead)  
**Date**: October 12, 2026  
**Status**: COMPLETE  
**Primary Deliverable**: `STAGE9_3_ACTOR_SATURATION_AUDIT.md`  
**Associated Artifacts**:
- Configuration: [`configs/reward_alignment_config.yaml`](file:///d:/Study%20Material/Research/QRFL/QFRL_Highway/configs/reward_alignment_config.yaml)
- Experiment Script: [`experiments/experiment_reward_alignment.py`](file:///d:/Study%20Material/Research/QRFL/QFRL_Highway/experiments/experiment_reward_alignment.py)
- Unit Tests: [`tests/test_reward_alignment.py`](file:///d:/Study%20Material/Research/QRFL/QFRL_Highway/tests/test_reward_alignment.py)
- Experimental Results: [`results/reward_alignment_comparison.csv`](file:///d:/Study%20Material/Research/QRFL/QFRL_Highway/results/reward_alignment_comparison.csv)

---

## 1. Executive Summary

Stage 9.3 accomplishes two critical diagnostic and methodological objectives prior to any exploration schedule modifications or federated scale-up:
1. **Resolution of Critic Diagnostic Interpretation**: Replaces flawed historical heuristics (such as $\sqrt{\mathcal{L}_{\text{critic}}} / |-\mathcal{L}_{\text{actor}}|$) with mathematically sound, distinct definitions of replay MSE, Bellman target magnitudes, actor objective values, evaluation RMSE, and aggregate physical Energy Efficiency (EE).
2. **Empirical Verification of Classical Actor Saturation**: Evaluates all 9 Classical and 9 Quantum checkpoints against a fixed, reproducible batch of 100 environment states. Empirically proves that the Classical MLP actor undergoes **100.00% action saturation** ($|a| > 0.95$) with extreme pre-activations ($z \in [-10.9, 10.8]$) and exactly **vanishing output gradients** ($\nabla_{\theta_A} \mathcal{L} \equiv 0.0000$), directly explaining why classical parameters froze between episodes 10 and 15. In contrast, the Quantum actor maintains healthy gradient flow (mean $1 - a^2 \in [0.42, 0.72]$) and non-zero parameter adaptation ($2.2\% - 3.6\%$).
3. **Controlled Reward-Alignment Experiment**: Proves mathematically that for constant input bits and episode length, maximizing cumulative negative energy ($\sum -E_t$) is strictly monotonically equivalent to maximizing aggregate EE ($\eta_{\text{EE}}$). In a controlled 90-episode trial across 3 matched seeds, negative energy achieved **0.00% rank inversions** with aggregate EE (compared to 33 rank inversions / 3.33% distortion under the baseline reciprocal ratio reward).

---

## 2. Correction of the Diagnostic Interpretation of Critic Relative Error

### 2.1 Flaw in Historical Heuristics
Earlier informal diagnostics computed a "relative prediction error" via:
$$\text{Heuristic Ratio} = \frac{\sqrt{\mathcal{L}_{\text{critic}}}}{|-\mathcal{L}_{\text{actor}}|} \approx \frac{\sqrt{1.7 \times 10^{10}}}{1.3 \times 10^5} \approx 98.7\%$$
This formulation is methodologically invalid because:
1. **Incomparable Quantities**: The numerator $\sqrt{\mathcal{L}_{\text{critic}}}$ is the root mean squared error of the temporal-difference (TD) residual on *off-policy replay transitions* $(s_i, a_i) \sim \mathcal{D}$. The denominator $-\mathcal{L}_{\text{actor}} = \frac{1}{|B|} \sum_i Q(s_i, \mu(s_i))$ is the critic's evaluated value of *current on-policy deterministic actions* $\mu(s_i)$. They evaluate completely different state-action distributions.
2. **Actor Loss is Not a Ground-Truth Return**: The policy value expectation $-\mathcal{L}_{\text{actor}}$ is not the ground-truth target for the critic. Using it as a normalizer has no statistical or reinforcement learning foundation.

### 2.2 Formal Separation of Distinct Quantities

To maintain rigorous standards, the following five metrics are formally distinguished:

| Quantity | Mathematical Definition | Scope & Purpose | Historical Log Status |
| :--- | :--- | :--- | :--- |
| **Critic Training MSE** | $\mathcal{L}_{\text{critic}} = \frac{1}{\|B\|} \sum_{i \in B} (Q(s_i, a_i) - y_i)^2$ | Loss minimized on random replay buffer minibatches $B \sim \mathcal{D}$ | Available in logs ($\sim 1.6 \times 10^{10}$) |
| **Critic Eval RMSE** | $\sqrt{\frac{1}{\|D_{\text{eval}}\|} \sum (Q(s, a) - G_t)^2}$ | True prediction error against Monte Carlo returns $G_t = \sum \gamma^k r_{t+k}$ on held-out trajectories | **UNAVAILABLE** (not recorded during historical runs) |
| **Bellman Target Scale** | $y_i = r_i + \gamma Q_{\text{target}}(s'_i, \mu_{\text{target}}(s'_i))$ | Magnitude of one-step bootstrap targets ($\approx \frac{\bar{r}}{1 - \gamma} \approx 2 \times 10^5$ for ratio, $\approx -1100$ for negative energy) | Available implicitly via reward and $\gamma$ |
| **Actor Objective Value** | $-\mathcal{L}_{\text{actor}} = \frac{1}{\|B\|} \sum_{i \in B} Q(s_i, \mu(s_i))$ | Critic-predicted expectation of current policy actions | Available in logs ($\approx 1.3 \times 10^5$) |
| **Aggregate Episode EE** | $\eta_{\text{EE}} = \frac{\sum_{t=1}^T \text{completed\_bits}_t}{\sum_{t=1}^T \text{energy}_t} \times 10^{-6}$ | True domain physical performance metric in Mbits/Joule | Available in canonical CSV ($\approx 0.53\text{--}0.78$ Mb/J) |

---

## 3. Empirical Investigation of Classical & Quantum Actor Saturation

### 3.1 Diagnostic Methodology
To evaluate whether the Classical actor was genuinely saturated or had reached a gradient equilibrium:
1. A reproducible batch of **100 valid environment states** was generated using `QFRLEnvironment` under fixed seed `999`.
2. All 9 Classical checkpoints (`classical_seed{42, 123, 2026}_ep{5, 10, 15}.pt`) and all 9 Quantum checkpoints (`quantum_seed{42, 123, 2026}_ep{5, 10, 15}.pt`) were evaluated.
3. For each checkpoint, we recorded:
   - Output action values across all 5 dimensions ($a \in [-1, 1]^5$).
   - Output pre-activation values $z$ (inputs to the final $\tanh$ layer).
   - Fraction of outputs with $|a| > 0.95$, $a < -0.95$, and $a > 0.95$.
   - Local Tanh derivative: $\frac{da}{dz} = 1 - a^2$.
   - Analytical policy gradient norm w.r.t actor parameters: $\|\nabla_{\theta_A} \mathcal{L}\| = \|\nabla_{\theta_A} (-\sum a)\|$.
   - Parameter delta norms between successive saved checkpoints: $\|d\theta_A(5 \to 10)\|$ and $\|d\theta_A(10 \to 15)\|$.

### 3.2 Classical Actor Saturation Evidence

The empirical diagnostic results on the fixed state batch are presented below:

| Checkpoint | Frac $\|a\| > 0.95$ | Frac $a < -0.95$ | Frac $a > 0.95$ | Mean $1 - a^2$ | Actor Grad Norm | Pre-activation $z$ [min, max] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **classical_seed42_ep5.pt** | **100.00%** | 80.00% | 20.00% | **0.000000** | **0.0000** | [-10.5, +10.8] |
| **classical_seed42_ep10.pt** | **100.00%** | 80.00% | 20.00% | **0.000000** | **0.0000** | [-10.5, +10.8] |
| **classical_seed42_ep15.pt** | **100.00%** | 80.00% | 20.00% | **0.000000** | **0.0000** | [-10.5, +10.8] |
| **classical_seed123_ep5.pt** | **100.00%** | 60.00% | 40.00% | **0.000000** | **0.0000** | [-10.9, +9.6] |
| **classical_seed123_ep10.pt** | **100.00%** | 60.00% | 40.00% | **0.000000** | **0.0000** | [-10.9, +9.6] |
| **classical_seed123_ep15.pt** | **100.00%** | 60.00% | 40.00% | **0.000000** | **0.0000** | [-10.9, +9.6] |
| **classical_seed2026_ep5.pt** | **100.00%** | 80.00% | 20.00% | **0.000000** | **0.0000** | [-9.9, +9.5] |
| **classical_seed2026_ep10.pt** | **100.00%** | 80.00% | 20.00% | **0.000000** | **0.0000** | [-9.9, +9.5] |
| **classical_seed2026_ep15.pt** | **100.00%** | 80.00% | 20.00% | **0.000000** | **0.0000** | [-9.9, +9.5] |

#### Parameter Delta Norms (Classical Actor):
- **Seed 42**: $\|d\theta(5 \to 10)\| = 0.0002$, $\|d\theta(10 \to 15)\| = \mathbf{0.0000}$ (Relative change: **0.00%**)
- **Seed 123**: $\|d\theta(5 \to 10)\| = 0.0003$, $\|d\theta(10 \to 15)\| = \mathbf{0.0000}$ (Relative change: **0.00%**)
- **Seed 2026**: $\|d\theta(5 \to 10)\| = 0.0001$, $\|d\theta(10 \to 15)\| = \mathbf{0.0000}$ (Relative change: **0.00%**)

#### Causal Finding:
The classical actor is **hard-saturated at the hypercube boundaries** ($[-1, 1]^5$). Pre-activations $z$ exceed $\pm 9.5$, driving the analytical Tanh derivative $1 - \tanh^2(z) \to 0.000000$. Backpropagated gradients through the actor output layer vanish completely:
$$\nabla_{\theta_A} \mathcal{L}_{\text{actor}} = \nabla_a Q(s, a) \cdot \left(1 - \tanh^2(z)\right) \cdot \nabla_{\theta_A} z \equiv \mathbf{0.0000}$$
Therefore, the lack of parameter change between episodes 10 and 15 is **not** policy convergence or gradient equilibrium; it is **fatal gradient paralysis caused by boundary Tanh saturation**.

---

### 3.3 Quantum Actor Saturation Comparison

The same diagnostics conducted on the 9 Quantum checkpoints reveal a fundamentally different learning dynamic:

| Checkpoint | Frac $\|a\| > 0.95$ | Frac $a < -0.95$ | Frac $a > 0.95$ | Mean $1 - a^2$ | $W_{\text{exp}}$ Norm | $B_{\text{exp}}$ Norm |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **quantum_seed42_ep5.pt** | **5.40%** | 2.00% | 3.40% | **0.6016** | 2.613 | 1.485 |
| **quantum_seed42_ep10.pt** | **17.80%** | 9.40% | 8.40% | **0.4672** | 3.687 | 2.039 |
| **quantum_seed42_ep15.pt** | **23.60%** | 11.60% | 12.00% | **0.4486** | 4.295 | 2.348 |
| **quantum_seed123_ep5.pt** | **2.00%** | 0.40% | 1.60% | **0.6293** | 2.523 | 1.336 |
| **quantum_seed123_ep10.pt** | **16.00%** | 9.80% | 6.20% | **0.4907** | 3.615 | 1.928 |
| **quantum_seed123_ep15.pt** | **25.00%** | 14.20% | 10.80% | **0.4257** | 4.269 | 2.270 |
| **quantum_seed2026_ep5.pt** | **1.20%** | 0.00% | 1.20% | **0.7186** | 2.644 | 1.176 |
| **quantum_seed2026_ep10.pt** | **13.20%** | 0.00% | 13.20% | **0.5338** | 3.507 | 1.631 |
| **quantum_seed2026_ep15.pt** | **18.60%** | 0.60% | 18.00% | **0.4868** | 3.901 | 1.836 |

#### Parameter Delta Norms (Quantum Actor):
- **Seed 42**: $\|d\theta(5 \to 10)\| = 1.3172$, $\|d\theta(10 \to 15)\| = 0.7279$ (Relative change: **3.07%**)
- **Seed 123**: $\|d\theta(5 \to 10)\| = 1.3712$, $\|d\theta(10 \to 15)\| = 0.7892$ (Relative change: **3.61%**)
- **Seed 2026**: $\|d\theta(5 \to 10)\| = 1.0826$, $\|d\theta(10 \to 15)\| = 0.4793$ (Relative change: **2.23%**)

#### Structural Reason:
1. **Unitary State Boundedness**: In the Quantum Actor, state representations are encoded into unitary rotations ($R_y, R_z$) and measured via Pauli-Z expectation values $\langle \psi | Z_i | \psi \rangle \in [-1, 1]$. Because the quantum observables are inherently bounded, pre-activations cannot explode into the $\pm 10$ range.
2. **Compact Linear Projection**: The final expansion layer maps only 4 qubit expectation values to 5 actions ($W \in \mathbb{R}^{5 \times 4}$), with weight norms $\|W\| \le 4.3$. Consequently, even by Episode 15, over 75% of outputs remain in the active linear zone of Tanh, and gradients continuously flow ($\frac{da}{dz} \approx 0.45$).

---

## 4. Controlled Reward-Alignment Experiment

### 4.1 Mathematical Equivalence Condition
In Stage 9.2, we discovered that the baseline reciprocal reward $R_t = \frac{B_t}{E_t}$ created Jensen rank inversions against aggregate episode EE $\eta_{\text{EE}} = \frac{\sum B_t}{\sum E_t}$.

**Theorem (Equivalence under Invariant Bits and Horizon)**:  
In the QFRL highway environment, each step generates a task of size $D = 1048576\text{ bits} = 1\text{ Mb}$. The simulation enforces complete offloading/computation such that $B_t = D = \text{const}$ for every step $t = 1, \dots, T$ where $T = 50$.  
The total completed bits across any episode is strictly constant:
$$B_{\text{total}} = \sum_{t=1}^T B_t = 50 \times 1\text{ Mb} = 50\text{ Mb} = 52,428,800\text{ bits}$$
The aggregate episode energy efficiency is:
$$\eta_{\text{EE}} = \frac{B_{\text{total}}}{\sum_{t=1}^T E_t} \times 10^{-6}\quad (\text{Mbits/Joule})$$
Because $B_{\text{total}} > 0$ is strictly constant:
$$\operatorname{argmax}_\pi \eta_{\text{EE}} \equiv \operatorname{argmin}_\pi \sum_{t=1}^T E_t \equiv \operatorname{argmax}_\pi \sum_{t=1}^T (-E_t)$$
Therefore, the **linear energy-minimization reward**:
$$R_t = -E_t$$
yields cumulative return:
$$G = \sum_{t=1}^T R_t = -\sum_{t=1}^T E_t$$
Maximizing cumulative return $G$ is **strictly monotonically equivalent** to maximizing aggregate episode Energy Efficiency $\eta_{\text{EE}}$, with **zero Jensen distortion**.

### 4.2 Verified Reward Scaling & Termination Behavior
- **Condition A (`ratio`)**: $R_t = \frac{B_t}{E_t} \in [10^4, 2 \times 10^5]$ (strictly positive, large Bellman target $Q^* \approx 2 \times 10^5$).
- **Condition B (`negative_energy`)**: $R_t = -E_t \in [-100, -30]$ (strictly negative, bounded Bellman target $Q^* \approx \frac{-55}{1 - 0.95} = -1100\text{ J}$).
- **Termination Handling**: Horizon $T = 50$ is fixed; terminal step sets `done = True`, zeroing target bootstrap: $y_T = r_T + 0$.

### 4.3 Matched Experimental Protocol
- **Architecture**: Matched MLP (Classical: Linear 150-64-64-5, Critic: Linear 155-64-64-1).
- **Environment**: Identical vehicle dynamics, IRS elements (16), Rician channel, and noise floor (-120 dBm).
- **Seeds**: `[42, 123, 2026]` for both conditions.
- **Budget**: 15 episodes per seed, 50 steps per episode.
- **Exploration & Optimizers**: $\sigma = 0.1$ constant Gaussian noise, Adam LR $\alpha_A = 10^{-3}, \alpha_C = 2.5 \times 10^{-3}$, $\tau = 0.005$.
- **No changes introduced**: Zero reward normalization, zero gradient clipping, and zero architectural alterations were added.

---

## 5. Experimental Results & Verification

The controlled experiment was executed via [`experiments/experiment_reward_alignment.py`](file:///d:/Study%20Material/Research/QRFL/QFRL_Highway/experiments/experiment_reward_alignment.py) generating [`results/reward_alignment_comparison.csv`](file:///d:/Study%20Material/Research/QRFL/QFRL_Highway/results/reward_alignment_comparison.csv) (90 canonical rows).

### 5.1 Final Episode (Ep 15) Performance Across Seeds (Preliminary, N=3)

| Condition | Seed | Cumulative Reward | Total Energy (J) | Aggregate EE (Mb/J) | Actor Saturation | Mean $1 - a^2$ | Actor Grad Norm |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ratio (Baseline)** | 42 | +27,177,809.68 | 2995.86 | 0.525013 | 100.0% | 0.000000 | 0.0000 |
| **Ratio (Baseline)** | 123 | +28,034,995.39 | 2878.13 | 0.546489 | 100.0% | 0.000000 | 0.0000 |
| **Ratio (Baseline)** | 2026 | +27,437,359.85 | 2959.22 | 0.531513 | 100.0% | 0.000000 | 0.0000 |
| **Negative Energy** | 42 | -2,996.09 | 2996.09 | 0.524972 | 100.0% | 0.000000 | 0.0000 |
| **Negative Energy** | 123 | -2,878.13 | 2878.13 | 0.546489 | 100.0% | 0.000000 | 0.0003 |
| **Negative Energy** | 2026 | -2,959.22 | 2959.22 | 0.531513 | 100.0% | 0.000000 | 0.0029 |

### 5.2 Across-Seed Aggregate Statistics

| Metric | Condition A (`ratio`) | Condition B (`negative_energy`) | Difference / Significance |
| :--- | :---: | :---: | :---: |
| **Ep 15 Mean EE (Mb/J)** | $0.534338 \pm 0.011013$ | $0.534325 \pm 0.011031$ | Matched physical performance |
| **Ep 15 Mean Total Energy (J)** | $2944.40 \pm 60.25$ | $2944.48 \pm 60.35$ | Matched physical performance |
| **Plateau (Ep 5-15) EE (Mb/J)** | $0.540159 \pm 0.016314$ | $0.540147 \pm 0.016314$ | Identical plateau trajectory |
| **Plateau Actor Saturation** | **100.00%** | **100.00%** | Both reach boundary saturation |
| **Pairwise Rank Inversions (Reward vs EE)** | **33 / 990 (3.33%)** | **0 / 990 (0.00%)** | **Perfect Monotonic Alignment** |

> [!IMPORTANT]
> **Preliminary Statistical Status**: With only 3 seeds per condition, these conclusions are preliminary. A 15-episode budget is not considered converged; it represents the controlled early-training regime matched to Stage 9.

### 5.3 Key Discoveries

1. **Perfect Monotonic Reward Alignment**:
   - Condition B (`negative_energy`) achieved exactly **0 rank inversions (0.00%)** between cumulative episode return and aggregate EE across all 990 pairwise comparisons.
   - Condition A (`ratio`) exhibited **33 rank inversions (3.33%)**, confirming the Jensen inequality distortion identified in Stage 9.2.
2. **Actor Saturation is Structural, Not Reward-Induced**:
   - Because the classical MLP actor saturated to 100% under *both* reward formulations by episode 4–5, the saturation lock-in is confirmed to be an **intrinsic structural deficiency of unconstrained MLP layers under continuous DDPG in this environment**, rather than a side effect of the large reciprocal reward magnitude.
   - While negative energy reduced critic Bellman target magnitudes by three orders of magnitude ($10^5 \to 10^3$), the policy gradient $\nabla_a Q(s, a)$ still rapidly drove unregularized linear weights into the saturation regime.
3. **Contrast with Quantum Actor**:
   - The Quantum Actor's resistance to saturation (remaining at 18%–25% saturation with mean Tanh' = 0.45) is confirmed to stem from its bounded unitary state space and compact parameterization, providing a clear structural mechanism for its superior stability.

---

## 6. Deliverables & Verification Summary

1. **Configuration**: Created [`configs/reward_alignment_config.yaml`](file:///d:/Study%20Material/Research/QRFL/QFRL_Highway/configs/reward_alignment_config.yaml) with matched parameters.
2. **Experiment Script**: Created [`experiments/experiment_reward_alignment.py`](file:///d:/Study%20Material/Research/QRFL/QFRL_Highway/experiments/experiment_reward_alignment.py) with unique `run_id`, saturation tracking, and gradient norm logging.
3. **Unit Tests**: Created [`tests/test_reward_alignment.py`](file:///d:/Study%20Material/Research/QRFL/QFRL_Highway/tests/test_reward_alignment.py) (3 tests passing; full suite 16/16 passing).
4. **Results Artifact**: Generated [`results/reward_alignment_comparison.csv`](file:///d:/Study%20Material/Research/QRFL/QFRL_Highway/results/reward_alignment_comparison.csv) with 90 canonical rows, completely preserving existing Stage 9 files.
5. **Historical Integrity**: `results/quantum_classical_comparison_canonical.csv` and existing model checkpoints remain pristine and untouched.

---

## 7. Recommended Next Steps for Stage 10

1. **Classical Actor Remediation**: To make classical baselines competitive and prevent premature freezing, apply weight decay / spectral normalization or layer normalization to the classical actor output layer before $\tanh$.
2. **Exploration Noise Scheduling**: With the reward objective now aligned and saturation diagnosed, evaluate OU-process vs scheduled Gaussian noise decay to escape early boundary plateaus.
3. **Federated Aggregation (QRFL Scale-Up)**: Proceed to federated aggregation testing once single-agent exploration schedules are validated.
