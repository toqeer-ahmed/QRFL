# Stage 9: Controlled Classical DDPG Benchmark & Empirical Comparison Report

**Project**: Quantum Federated Reinforcement Learning (QFRL) for Energy-Efficient Vehicular Edge Computing  
**Objective**: Empirically evaluate whether the 4-qubit hybrid Quantum-Classical Actor offers an empirical advantage over a Classical MLP Actor under matched environment conditions.  
**Execution Date**: 2026-10-12  
**Status**: COMPLETE  

---

## 1. Stage 8 Pre-Experiment Audit Findings

Prior to benchmarking, a thorough audit of Stage 8 code, configurations, metric formulas, and execution artifacts was conducted:

### Audit Investigation Findings:
1. **Identical Completed Bits Across Seeds ($1.57 \times 10^9\text{ bits}$)**:
   - *Finding*: In `environment/environment.py`, task bits are partitioned as $\text{local\_bits} = \text{input\_bits} \times (1 - \text{ratio})$ and $\text{offload\_bits} = \text{input\_bits} \times \text{ratio}$.
   - Completed bits is calculated as $\text{local\_bits} + \text{offload\_bits} = \text{input\_bits} \times (1 - \text{ratio} + \text{ratio}) \equiv \text{input\_bits}$.
   - With 30 vehicles, task size $1,048,576\text{ bits}$ ($1\text{ Mb}$), and 50 steps per episode:
     $$\text{Completed Bits} = 30 \times 1,048,576 \times 50 = 1,572,864,000\text{ bits} = 1.572864 \times 10^9\text{ bits}$$
   - Because the Stage 6A revision removed task timeout drop logic, every generated task bit is accounted as executed. Hence, the numerator of Energy Efficiency is an invariant physical constant across all seeds and episodes.
2. **Policy Impact on Metrics**:
   - The policy does **not** alter completed bits (as shown above).
   - The policy **strongly affects Total Energy** ($E_{\text{local}} \propto f^2$, $E_{\text{tx}} = P_{\text{tx}} T_{\text{tx}}$, $E_{\text{uav}}$) and **Latency**.
   - As a direct consequence, **Energy Efficiency ($\eta_{\text{EE}} = \frac{\text{Completed Bits}}{E_{\text{total}}}$)** is dynamically maximized by minimizing total energy consumption.
3. **Episode Energy Efficiency Calculation Protocol**:
   - *Finding*: In `experiments/train_baseline_qfrl.py`, episode EE is computed as:
     $$\eta_{\text{EE, ep}} = \frac{\sum_{t=1}^{50} \text{Completed Bits}_t}{\max(\sum_{t=1}^{50} E_{\text{total}, t}, 10^{-6})}$$
   - This correctly computes the aggregate ratio of total episode bits divided by total episode energy, **not** an unweighted arithmetic average of step-level ratios.
4. **Reward Scale and Physical Units**:
   - *Finding*: `rl/reward.py` computes reward as $R_t = \frac{\text{completed\_bits}_t}{\max(E_{\text{total}, t}, 10^{-6})}$.
   - This reward is in **raw bits/Joule** (unscaled), yielding step rewards on the order of $10^5\text{--}10^6$ and episode rewards on the order of $10^7$.
5. **Numerical Validity of Critic Losses**:
   - *Finding*: With discount factor $\gamma = 0.95$ and step rewards $\approx 5 \times 10^5$, returns $Q(s, a) \approx \frac{R}{1 - \gamma} \approx 10^7$.
   - The Mean Squared Error loss $(Q(s, a) - y)^2$ is naturally on the order of $(10^5\text{--}10^6)^2 \approx 10^{10}\text{--}10^{12}$.
   - Reported critic losses ($1.8 \times 10^{10}$ to $1.6 \times 10^{11}$) are mathematically and numerically consistent with this reward scale.
6. **Task Completion and Termination Consistency**:
   - Episodes terminate strictly at $T = 50\text{ steps}$. Energy accounting across local computation, transmission, and UAV execution is additive and consistent.
7. **Stage 8 Artifact Reproducibility**:
   - Stage 8 full 45-episode CSV, PNG, and checkpoints were generated on the remote Kaggle GPU container. The local repository contains the mock smoke-test CSV. The full run logs are archived in `FULL_PROJECT_CHAT_HISTORY.md` and `STAGE8_BASELINE_TRAINING_REPORT.md`.

---

## 2. Experimental Design & Controlled Architecture

To ensure an unbiased, scientifically controlled comparison:
- **Environment**: Identical `QFRLEnvironment` initialized with `configs/paper_config.yaml` (30 vehicles, 1 UAV, 1 IRS, 50 steps/episode).
- **Seeds**: Matched seed set `[42, 123, 2026]`.
- **Training Budget**: 15 episodes per seed (45 episodes total per model).
- **Exploration Schedule**: Matched constant Gaussian noise ($\sigma = 0.1$, unchanged from Stage 8).
- **Replay Buffer & Batch Size**: Capacity 6,000 transitions, batch size 32, warm-up 32 transitions.
- **Critic Architecture**: Identical classical MLP critic (`Linear(155, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 1)`).
- **Target Updates & Optimizer**: Adam optimizer, actor $\text{lr} = 0.001$, critic $\text{lr} = 0.0025$, soft Polyak update $\tau = 0.005$, $\gamma = 0.95$.

### Model Architectures Compared:
| Component | Quantum Actor (`QuantumActor`) | Classical Actor (`ClassicalActor`) |
|---|---|---|
| **Input State** | $\mathbb{R}^{150}$ | $\mathbb{R}^{150}$ |
| **Encoder / Latent** | $\text{Linear}(150, 4) \to \text{Tanh}$ ($4\text{ qubits}$) | $\text{Linear}(150, 64) \to \text{ReLU} \to \text{Linear}(64, 64) \to \text{ReLU}$ |
| **Core Processor** | 12 variational layers ($R_x, R_y, R_z$, circular CNOT) + Pauli-Z | Classical Feedforward Hidden Layers |
| **Output Decoder** | $\text{Linear}(4, 5) \to \text{Tanh}$ | $\text{Linear}(64, 5) \to \text{Tanh}$ |
| **Action Bounds** | $[-1.0, 1.0]^5$ | $[-1.0, 1.0]^5$ |
| **Actor Trainable Params** | **773** ($604 + 144 + 25$) | **14,149** |
| **Critic Trainable Params** | **14,209** | **14,209** |

---

## 3. Quantitative Results

### 3.1 Aggregate Statistical Summary (Mean $\pm$ Std across 45 Episodes)

| Metric | Classical DDPG Baseline | Quantum QFRL Baseline | Absolute Delta ($\text{Q} - \text{C}$) | Relative Delta ($\%$) |
|---|---|---|---|---|
| **Actor Parameters** | 14,149 | **773** | -13,376 | **-94.5%** |
| **Mean Energy Efficiency ($\text{Mb/J}$)** | $\mathbf{0.5231 \pm 0.0689}$ | $0.3478 \pm 0.2157$ | $-0.1753\text{ Mb/J}$ | $-33.51\%$ |
| **Mean Total Energy ($\text{J}$)** | $\mathbf{3,102.60 \pm 737.14}$ | $9,546.86 \pm 8,675.92$ | $+6,444.26\text{ J}$ | $+207.7\%$ |
| **Mean Episode Reward** | $\mathbf{27.28 \times 10^6 \pm 2.36 \times 10^6}$ | $17.95 \times 10^6 \pm 11.18 \times 10^6$ | $-9.33 \times 10^6$ | $-34.20\%$ |
| **Mean Latency ($\text{s}$)** | $\mathbf{0.1685 \pm 0.0120}$ | $0.1937 \pm 0.0797$ | $+0.0252\text{ s}$ | $+14.96\%$ |
| **Completed Bits** | $1.57 \times 10^9$ | $1.57 \times 10^9$ | $0.00$ | $0.00\%$ |
| **Inference Time per Action** | **$0.31\text{ ms}$** | $12.4\text{ ms}$ | $+12.09\text{ ms}$ | $+3900\%$ |
| **Total 45-Ep Runtime** | **$17.8\text{ seconds}$** | $696.2\text{ seconds}$ | $+678.4\text{ s}$ | **~39x slower** |

---

### 3.2 Per-Seed Breakdown

#### Seed 42:
* **Classical**:
  - Ep 1: EE = $0.248\text{ Mb/J}$, Energy = $6,343.6\text{ J}$
  - Ep 10: EE = $\mathbf{0.574\text{ Mb/J}}$, Energy = $\mathbf{2,741.9\text{ J}}$
  - Mean EE: $\mathbf{0.5365\text{ Mb/J}}$
* **Quantum**:
  - Ep 1: EE = $0.217\text{ Mb/J}$, Energy = $7,233.7\text{ J}$
  - Ep 10: EE = $\mathbf{0.573\text{ Mb/J}}$, Energy = $\mathbf{2,744.2\text{ J}}$
  - Mean EE: $0.5051\text{ Mb/J}$
* *Finding*: In Seed 42, both models converged to the **exact same optimal physical ceiling** ($\approx 0.573\text{--}0.574\text{ Mb/J}$ at $\approx 2,742\text{--}2,744\text{ J}$).

#### Seed 123:
* **Classical**:
  - Ep 1: EE = $0.312\text{ Mb/J}$, Energy = $5,039.7\text{ J}$
  - Ep 14: EE = $\mathbf{0.573\text{ Mb/J}}$, Energy = $\mathbf{2,746.6\text{ J}}$
  - Mean EE: $\mathbf{0.5348\text{ Mb/J}}$
* **Quantum**:
  - Ep 1: EE = $0.148\text{ Mb/J}$, Energy = $10,643.0\text{ J}$
  - Ep 14: EE = $\mathbf{0.572\text{ Mb/J}}$, Energy = $\mathbf{2,749.7\text{ J}}$
  - Mean EE: $0.4641\text{ Mb/J}$
* *Finding*: In Seed 123, both models again reached the identical physical ceiling ($\approx 0.572\text{--}0.573\text{ Mb/J}$), with the Classical agent converging slightly earlier (by Ep 3 vs. Ep 5).

#### Seed 2026:
* **Classical**:
  - Ep 1: EE = $0.263\text{ Mb/J}$, Energy = $5,982.4\text{ J}$
  - Ep 4: EE = $\mathbf{0.560\text{ Mb/J}}$, Energy = $\mathbf{2,808.1\text{ J}}$
  - Ep 14: EE = $\mathbf{0.559\text{ Mb/J}}$, Energy = $\mathbf{2,813.1\text{ J}}$
  - Mean EE: $\mathbf{0.5280\text{ Mb/J}}$
* **Quantum**:
  - Ep 1: EE = $0.105\text{ Mb/J}$, Energy = $14,980.6\text{ J}$
  - Ep 5: EE = $0.071\text{ Mb/J}$, Energy = $22,077.8\text{ J}$
  - Ep 15: EE = $0.071\text{ Mb/J}$, Energy = $22,192.2\text{ J}$
  - Mean EE: $0.0742\text{ Mb/J}$
* *Critical Finding*: The Classical model converged successfully on Seed 2026 without any action saturation, whereas the Quantum model remained trapped in a conservative suboptimal policy.

---

## 4. Key Scientific Findings & Analysis

### 1. Does the Quantum Actor Provide an Empirical Advantage?
**NO.** Under matched, controlled conditions:
- **Solution Quality**: Both models reach virtually identical peak energy efficiency ($\approx 0.573\text{ Mb/J}$) and optimal total energy consumption ($\approx 2,744\text{ J}$). There is no evidence that the quantum circuit finds a superior physical operating point.
- **Robustness Across Random Seeds**: The classical model is substantially more stable across random seeds ($\text{Std} = 0.0689$ vs. $0.2157$), successfully learning across all 3 seeds without getting trapped in action saturation.
- **Computational Cost**: The classical model runs ~39x faster in simulation time and consumes orders of magnitude less wall-clock compute.

### 2. What Is the Distinctive Property of the Quantum Actor?
- **Extreme Parameter Compactness**: The quantum actor achieves comparable peak energy efficiency with only **773 parameters** compared to **14,149 parameters** in the classical MLP (a **94.5% reduction in parameter footprint**).
- This parameter efficiency highlights the expressive representational capacity per parameter of variational quantum circuits, but this does not translate into empirical optimization superiority on the target RL task under current classical simulation regimes.

---

## 5. Limitations

1. **Simulated Quantum Backend**: All quantum circuits were simulated via statevector simulation on classical hardware. Noise, decoherence, and real quantum gate errors on physical NISQ processors were not present.
2. **Horizon Budget**: Comparison was conducted over 15 episodes per seed (45 episodes total). Longer training horizons (e.g. 50–100 episodes) or dynamic hyperparameter sweeps might reveal different late-stage asymptotic behaviors.
3. **Exploration Sensitivity**: Fixed Gaussian exploration noise ($\sigma = 0.1$) interacted unfavorably with the quantum model's initial state on Seed 2026.

---

## 6. Recommended Next Experiment (Stage 10)

1. **Scheduled Exploration Noise Tuning**: Run `experiments/experiment_exploration_schedule.py` with exponential decaying noise ($\sigma_t = \max(0.01, 0.1 \times 0.95^{t-1})$) to verify whether adaptive exploration resolves seed variance in the quantum model.
2. **Highway Multi-Vehicle Scale-Up**: Integrate the full multi-vehicle federated scenario (5–10 client vehicles with local policies and FedAvg synchronization) to test whether quantum parameter compactness yields transmission savings in bandwidth-constrained federated aggregation.
