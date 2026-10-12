# STAGE 9.4 AUDIT REPORT: VERIFICATION OF ACTOR GRADIENTS AND REWARD DISCOUNTING

**Author**: Antigravity Autonomous Agent (Pair Programming with Research Lead)  
**Date**: October 12, 2026  
**Status**: COMPLETE  
**Primary Deliverable**: `STAGE9_4_GRADIENT_AND_DISCOUNT_AUDIT.md`  
**Associated Artifacts**:
- Unit Tests: [`tests/test_gradient_and_discount.py`](file:///d:/Study%20Material/Research/QRFL/QFRL_Highway/tests/test_gradient_and_discount.py)
- Diagnostic Scripts: `scratch/stage9_4_diagnostics.py`, `scratch/stage9_4_quantum_diagnostics.py`

---

## 1. Executive Summary & Classification of Claims

Stage 9.4 audits and rigorously tests the conclusions reached in Stage 9.3 using full-precision numerical diagnostics, real backward passes with checkpointed critics, and independent mathematical proofs.

Each primary conclusion is systematically evaluated and classified below:

| Finding / Conclusion | Audit Status | Diagnostic Evidence |
| :--- | :---: | :--- |
| **Float32 Classical Actor Gradient Vanishing** | **VERIFIED BY DIRECT MEASUREMENT** | Real backward pass with checkpoint critic yields **exactly $\|\nabla_{\theta_A} \mathcal{L}\| = 0.000000e+00$** across all 9 classical checkpoints. Float32 autograd evaluates $1 - a^2 = 0.0f$ because $|z| \ge 9.2$ rounds $\tanh(z)$ to $\pm 1.0f$. Analytically, $\text{sech}^2(z) \sim 10^{-9} < \frac{1}{2}\epsilon_{\text{mach}}$. |
| **Critic Isolation during Actor Step** | **VERIFIED BY DIRECT MEASUREMENT** | Critic receives non-zero autograd gradients during `actor_loss.backward()` ($\|\nabla_{\theta_C}\| \approx 8.3\times 10^3\text{--}6.5\times 10^4$), but `actor_optimizer.step()` contains only actor parameters and leaves critic parameters with **exactly $0.000000e+00$ delta**. Gradients are cleared by `critic_optimizer.zero_grad()` before the next critic update. |
| **Classical Checkpoint Bit-Exact Freeze** | **VERIFIED BY DIRECT MEASUREMENT** | Maximum absolute parameter delta between Episode 10 and 15 is **$0.00000000e+00$ (bit-exact zero)** across all 14,149 parameters for Seeds 42, 123, and 2026. |
| **Quantum Actor Gradient Preservation** | **VERIFIED BY DIRECT MEASUREMENT** | Real backward pass on Quantum checkpoints yields non-zero actor gradient norms ($7.56\times 10^2\text{--}1.54\times 10^4$), healthy Tanh derivatives (mean $0.388\text{--}0.663$), low saturation (only 1.4%–29.8% at $|a|>0.95$ and 0.0%–6.6% at $|a|>0.99$), and parameter steps of $\approx 5.0\times 10^{-3}$. |
| **Undiscounted Negative Energy Equivalence to EE** | **VERIFIED BY DIRECT MEASUREMENT** | When total task bits $\sum B_t$ and horizon $T$ are constant, undiscounted cumulative return $\sum_{t=0}^{T-1} -E_t = -E_{\text{total}}$ is strictly monotonic to aggregate EE $\eta_{\text{EE}} = \frac{B_{\text{total}}}{E_{\text{total}}}$, with **0 / 990 rank inversions (0.00%)**. |
| **Discounted DDPG Objective Equivalence to EE** | **NOT VERIFIED / THEORETICALLY NON-EQUIVALENT** | For $\gamma = 0.95 < 1$, discounted return $\sum_{t=0}^{T-1} \gamma^t (-E_t)$ weights early steps exponentially more heavily than late steps ($\gamma^{49} \approx 0.0769$). Counterexample proven: a sequence with higher total energy can yield a higher discounted return, introducing temporal preference. |
| **Architectural Limitation of Classical MLPs** | **SUPPORTED BUT NOT PROVEN** | The boundary lock-in occurred under both rewards in the tested unregularized continuous DDPG configuration (LR $10^{-3}$, Gaussian noise 0.1, no weight decay). However, whether regularized MLPs (e.g., spectral normalization, layer normalization, or smaller output weight scaling) could prevent saturation remains unproven. |
| **Policy Convergence at Episode 15** | **NOT VERIFIED (DISPROVEN)** | The parameter stagnation between Episodes 10 and 15 was **float32 gradient paralysis**, not mathematical convergence to a local or global optimum. |

---

## 2. Task 1: Full-Precision Classical & Quantum Actor Gradient Verification

### 2.1 Classical Actor Gradient Diagnostics
Each of the 9 saved classical checkpoints was loaded and evaluated on the fixed, reproducible batch of 100 on-distribution states from `QFRLEnvironment`. A real backward pass was executed using the actual critic:
$$\mathcal{L}_{\text{actor}} = -\frac{1}{100} \sum_{i=1}^{100} Q\left(s_i, \mu_{\theta_A}(s_i)\right)$$

#### Table 1: Full-Precision Classical Actor Diagnostics across Checkpoints
| Checkpoint | Saturation $\|a\|>0.95$ | Saturation $\|a\|>0.99$ | Tanh' $(1 - a^2)$ [min, median, mean, max] | Real Actor Grad Norm | Critic Grad Norm (Actor Backprop) | Actor $\Delta\theta$ after `step()` | Critic $\Delta\theta$ after `step()` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **classical_seed42_ep5.pt** | 100.00% | 100.00% | `[0.000e+0, 0.000e+0, 0.000e+0, 0.000e+0]` | **0.000000e+00** | $9.878\times 10^3$ | **0.000000e+00** | **0.000000e+00** |
| **classical_seed42_ep10.pt** | 100.00% | 100.00% | `[0.000e+0, 0.000e+0, 0.000e+0, 0.000e+0]` | **0.000000e+00** | $3.560\times 10^4$ | **0.000000e+00** | **0.000000e+00** |
| **classical_seed42_ep15.pt** | 100.00% | 100.00% | `[0.000e+0, 0.000e+0, 0.000e+0, 0.000e+0]` | **0.000000e+00** | $6.387\times 10^4$ | **0.000000e+00** | **0.000000e+00** |
| **classical_seed123_ep5.pt** | 100.00% | 100.00% | `[0.000e+0, 0.000e+0, 0.000e+0, 0.000e+0]` | **0.000000e+00** | $8.976\times 10^3$ | **0.000000e+00** | **0.000000e+00** |
| **classical_seed123_ep10.pt** | 100.00% | 100.00% | `[0.000e+0, 0.000e+0, 0.000e+0, 0.000e+0]` | **0.000000e+00** | $3.479\times 10^4$ | **0.000000e+00** | **0.000000e+00** |
| **classical_seed123_ep15.pt** | 100.00% | 100.00% | `[0.000e+0, 0.000e+0, 0.000e+0, 0.000e+0]` | **0.000000e+00** | $5.818\times 10^4$ | **0.000000e+00** | **0.000000e+00** |
| **classical_seed2026_ep5.pt** | 100.00% | 100.00% | `[0.000e+0, 0.000e+0, 0.000e+0, 0.000e+0]` | **0.000000e+00** | $8.363\times 10^3$ | **0.000000e+00** | **0.000000e+00** |
| **classical_seed2026_ep10.pt** | 100.00% | 100.00% | `[0.000e+0, 0.000e+0, 0.000e+0, 0.000e+0]` | **0.000000e+00** | $3.439\times 10^4$ | **0.000000e+00** | **0.000000e+00** |
| **classical_seed2026_ep15.pt** | 100.00% | 100.00% | `[0.000e+0, 0.000e+0, 0.000e+0, 0.000e+0]` | **0.000000e+00** | $5.878\times 10^4$ | **0.000000e+00** | **0.000000e+00** |

#### Per-Dimension Action and Pre-Activation Values:
Across all 100 environment states:
- **Seed 42**: $z \approx [-10.5, -9.6, -9.6, +10.8, -9.6]$, yielding $a = [-1.0, -1.0, -1.0, +1.0, -1.0]$ across all 5 dimensions (100% saturated at $\pm 1$).
- **Seed 123**: $z \approx [-9.6, -10.9, +9.6, +9.2, -9.6]$, yielding $a = [-1.0, -1.0, +1.0, +1.0, -1.0]$ (100% saturated at $\pm 1$).
- **Seed 2026**: $z \approx [-9.9, -9.3, +9.5, -9.3, -9.7]$, yielding $a = [-1.0, -1.0, +1.0, -1.0, -1.0]$ (100% saturated at $\pm 1$).

#### Checkpoint Parameter Comparison at Full Numerical Precision:
- **Seed 42**: $\max |\theta_{10} - \theta_5| = 4.16859984\times 10^{-5}$; $\max |\theta_{15} - \theta_{10}| = \mathbf{0.00000000e+00}$.
- **Seed 123**: $\max |\theta_{10} - \theta_5| = 1.13494694\times 10^{-4}$; $\max |\theta_{15} - \theta_{10}| = \mathbf{0.00000000e+00}$.
- **Seed 2026**: $\max |\theta_{10} - \theta_5| = 2.17556953\times 10^{-5}$; $\max |\theta_{15} - \theta_{10}| = \mathbf{0.00000000e+00}$.

> [!NOTE]
> **Analytical vs. Floating-Point Derivative**:  
> Analytically, $\text{sech}^2(z) = 4/(e^z + e^{-z})^2$. For $z \approx 9.5$, $\text{sech}^2(z) \approx 2.2\times 10^{-8}$.  
> In IEEE-754 single-precision `float32`, machine epsilon is $\epsilon_{\text{mach}} \approx 1.19\times 10^{-7}$. Because $|1 - \tanh(z)| < \epsilon_{\text{mach}}/2$, `torch.tanh(z)` rounds to *exact* $\pm 1.0f$, and autograd evaluates $1 - a^2 = 1.0f - (1.0f)^2 = 0.0f$.  
> Thus, the vanishing gradient is an **exact numerical float32 underflow**, confirmed by real backward passes.

---

### 2.2 Quantum Actor Gradient Diagnostics
The same diagnostics conducted on the 9 Quantum checkpoints demonstrate continuous, healthy gradient flow throughout training:

#### Table 2: Full-Precision Quantum Actor Diagnostics across Checkpoints
| Checkpoint | Saturation $\|a\|>0.95$ | Saturation $\|a\|>0.99$ | Tanh' $(1 - a^2)$ [min, median, mean, max] | Real Actor Grad Norm | Critic Grad Norm (Actor Backprop) | Actor $\Delta\theta$ after `step()` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **quantum_seed42_ep5.pt** | 3.40% | 0.00% | `[0.0445, 0.6367, 0.6119, 1.0000]` | **1.399e+03** | $1.056\times 10^4$ | $5.000\times 10^{-3}$ |
| **quantum_seed42_ep10.pt** | 19.00% | 2.60% | `[0.0070, 0.3961, 0.4580, 0.9999]` | **7.599e+03** | $4.095\times 10^4$ | $5.000\times 10^{-3}$ |
| **quantum_seed42_ep15.pt** | 22.00% | 6.20% | `[0.0030, 0.4402, 0.4606, 1.0000]` | **1.535e+04** | $6.494\times 10^4$ | $5.000\times 10^{-3}$ |
| **quantum_seed123_ep5.pt** | 1.40% | 0.00% | `[0.0531, 0.7051, 0.6632, 1.0000]` | **7.568e+02** | $5.872\times 10^3$ | $5.000\times 10^{-3}$ |
| **quantum_seed123_ep10.pt** | 12.80% | 0.80% | `[0.0149, 0.5550, 0.5414, 1.0000]` | **5.614e+03** | $2.575\times 10^4$ | $5.000\times 10^{-3}$ |
| **quantum_seed123_ep15.pt** | 29.80% | 6.60% | `[0.0020, 0.2634, 0.3882, 1.0000]` | **1.311e+04** | $4.582\times 10^4$ | $5.000\times 10^{-3}$ |
| **quantum_seed2026_ep5.pt** | 4.60% | 0.20% | `[0.0154, 0.6567, 0.6247, 1.0000]` | **1.008e+03** | $7.705\times 10^3$ | $5.000\times 10^{-3}$ |
| **quantum_seed2026_ep10.pt** | 12.40% | 1.20% | `[0.0052, 0.5106, 0.5214, 1.0000]` | **2.326e+03** | $1.632\times 10^4$ | $5.000\times 10^{-3}$ |
| **quantum_seed2026_ep15.pt** | 18.80% | 6.00% | `[0.0018, 0.4282, 0.4713, 1.0000]` | **2.706e+03** | $2.209\times 10^4$ | $5.000\times 10^{-3}$ |

---

## 3. Task 2: Reward Discounting & Target Calculation Audit

### 3.1 DDPG Target Formulation & Termination Handling
- **Implemented Discount Factor**: $\gamma = 0.95$ (specified in `configs/paper_config.yaml` and initialized in `ClassicalDDPGAgent` and `DDPGAgent`).
- **Terminal State Masking**:
  $$\text{target\_q} = r_t + (1 - d_t) \cdot \gamma \cdot Q_{\text{target}}(s_{t+1}, \mu_{\text{target}}(s_{t+1}))$$
  When $d_t = 1$, $(1 - d_t) = 0$, explicitly zeroing out the bootstrapped future value.
- **Episode Horizon Handling**: In `environment/environment.py`, `done` is set to `True` at step $T = 50$ (`self.current_step >= self.max_steps`). Thus, step 50 is treated as terminal.

### 3.2 Difference Between Discounted Returns and Undiscounted Total Energy

Let $E_t \ge 0$ denote step energy consumption, and $R_t = -E_t$ denote the linear negative-energy reward.
- **Undiscounted Total Energy Return**:
  $$G_{\text{undisc}} = \sum_{t=0}^{T-1} R_t = -\sum_{t=0}^{T-1} E_t = -E_{\text{total}}$$
  Because total completed bits $B_{\text{total}} = 50\text{ Mb}$ is constant, maximizing $G_{\text{undisc}}$ is strictly equivalent to maximizing aggregate Energy Efficiency $\eta_{\text{EE}} = \frac{B_{\text{total}}}{E_{\text{total}}}$.
- **Discounted Return**:
  $$G_{\text{disc}} = \sum_{t=0}^{T-1} \gamma^t R_t = -\sum_{t=0}^{T-1} \gamma^t E_t \quad (\gamma = 0.95)$$
  Because $\gamma^t$ decays geometrically ($1.0 \to 0.95 \to \dots \to 0.0769$), the discounted objective creates a temporal bias toward **deferring energy consumption to later steps**.

#### Deterministic Counterexample (Rank Inversion under Discounting):
Consider two 2-step energy trajectories with $\gamma = 0.95$:
- **Trajectory A (Front-loaded energy)**: $E_A = [60.0\text{ J}, 10.0\text{ J}]$
  - Total Energy: $60.0 + 10.0 = \mathbf{70.0\text{ J}}$
  - Aggregate EE: $2\text{ Mb} / 70.0\text{ J} = \mathbf{0.02857\text{ Mb/J}}$ (Superior)
  - Discounted Return: $-60.0 - 0.95 \times 10.0 = \mathbf{-69.50\text{ J}}$
- **Trajectory B (Back-loaded energy)**: $E_B = [10.0\text{ J}, 62.0\text{ J}]$
  - Total Energy: $10.0 + 62.0 = \mathbf{72.0\text{ J}}$ (Inferior)
  - Aggregate EE: $2\text{ Mb} / 72.0\text{ J} = \mathbf{0.02778\text{ Mb/J}}$
  - Discounted Return: $-10.0 - 0.95 \times 62.0 = \mathbf{-68.90\text{ J}}$ (Higher Return: $-68.90 > -69.50$)

Trajectory B has **higher total energy** (inferior physical efficiency), but yields a **higher discounted return**.  
Therefore, stating that DDPG with $\gamma = 0.95$ optimizes aggregate EE is a theoretical approximation, not an exact equivalence.

### 3.3 Reward Numerical Scale Differences
| Reward Formulation | Step Reward Scale | Discounted Target Scale $Q^*$ | Critic MSE Loss Scale |
| :--- | :---: | :---: | :---: |
| **Condition A (`ratio`)** | $+10,000\text{ to }+200,000\text{ bits/J}$ | $\approx +2\times 10^5\text{ to }+10^6$ | $10^{10}\text{ to }10^{11}$ |
| **Condition B (`negative_energy`)** | $-30\text{ to }-100\text{ J}$ | $\approx -1,100\text{ J}$ | $10^2\text{ to }10^4$ |

The ratio reward produces Bellman targets and critic losses that are **$3$ to $5$ orders of magnitude larger** than linear negative energy.

---

## 4. Corrections to Stage 9.3 Interpretations

1. **Correction on "Zero Gradients"**:
   - *Previous statement*: "Analytical gradient norm is 0.0000 due to $1 - a^2 = 0$."
   - *Audited precision*: In float32 autograd, actor gradients are **strictly $0.000000e+00$** due to IEEE-754 underflow of $\text{sech}^2(z)$ relative to 1.0 when $|z| \ge 9.2$. Analytically, gradients are $\sim 10^{-9}$, but in machine precision they are exactly zero.
2. **Correction on "Architectural Limitation"**:
   - *Previous statement*: "Actor saturation is an intrinsic architectural limitation of unconstrained MLP layers."
   - *Audited precision*: Saturation occurred in the tested unregularized MLP with Adam ($\alpha=10^{-3}$) and Gaussian noise. It is **supported but not proven** to be an inherent limitation of all classical architectures, as regularized MLPs (spectral normalization, layer normalization, weight decay) were not evaluated.
3. **Correction on "Exact Reward Equivalence"**:
   - *Previous statement*: "Minimizing total energy is equivalent to maximizing aggregate EE."
   - *Audited precision*: Strictly true for **undiscounted return** ($\gamma = 1$), but **not strictly equivalent for discounted return** ($\gamma = 0.95$). Discounting introduces a temporal weight decay ($\gamma^{49} \approx 0.0769$) that can invert policy preferences between front-loaded and back-loaded energy profiles.
4. **Correction on "Convergence"**:
   - Stagnation between Ep 10 and Ep 15 is **gradient paralysis**, not mathematical convergence to an optimal policy.

---

## 5. Verification Deliverables Summary

- **Unit Test Suite**: [`tests/test_gradient_and_discount.py`](file:///d:/Study%20Material/Research/QRFL/QFRL_Highway/tests/test_gradient_and_discount.py) (4 tests passing; full repository suite 20/20 passing).
- **All Historical Artifacts Preserved**: `results/quantum_classical_comparison_canonical.csv` and existing checkpoints remain untouched.
- **Git Commit**: Staged and committed locally.
