# PAPER RL FORMULATION EXTRACTION

## 1. State
- **State dimension**: Explicitly specified as 150.
- **State variables/features**: The paper mentions variables related to vehicles, channels, and environment positions, but the exact ordering and construction of the 150-dimensional vector is NOT SPECIFIED IN PAPER.
- **State scope (per-vehicle vs global)**: NOT SPECIFIED IN PAPER.

## 2. Action
- **Action dimension**: Explicitly specified as 5.
- **Action variables/mapping**: The paper lists optimization variables (task offloading, transmit power, IRS phase, UAV trajectory), but how they mathematically map to exactly 5 dimensions is NOT SPECIFIED IN PAPER. Action dimension = 5 is reported, but exact 5-dimensional action-vector mapping is not specified.

## 3. Optimization Variables
| Variable | Meaning | Mathematical notation | Where used | RL action? | Paper evidence |
|----------|---------|-----------------------|------------|------------|----------------|
| Task offloading | Decision to compute locally vs edge | NOT SPECIFIED IN PAPER | Computing/Energy logic | NOT SPECIFIED IN PAPER | Mentioned as optimization objective |
| Transmit power | Power used by VUEs to offload | NOT SPECIFIED IN PAPER | SNR/Rate calculation | NOT SPECIFIED IN PAPER | Mentioned as optimization objective |
| IRS phase | Reflection shifts at IRS | `Theta_tau` / `alpha_p` | Channel path loss | NOT SPECIFIED IN PAPER | Mentioned as optimization objective |
| UAV trajectory | Position/movement of UAV | `X_u` / `(l_u, m_u, H)` | Distance/Path loss | NOT SPECIFIED IN PAPER | Mentioned as optimization objective |
| CPU allocation | Local and edge clock speeds | `f_v`, `f_r`, `f_u` | Computation logic | NOT SPECIFIED IN PAPER | Model derivations |

## 4. Objective
- **Optimization Objective**: Energy Efficiency (conceptualized as completed task bits / total energy consumption).
- **RL Reward Function**: Paper does not explicitly provide a complete RL reward equation (e.g., whether it includes latency penalties or scaling constants). 

## 5. Reward
- **Reward definition**: NOT SPECIFIED IN PAPER (Only the overarching Energy Efficiency objective is defined, the exact step-by-step reward structure is not mathematically given in the available materials).

## 6. QFRL Architecture
- **Actor/Critic Structure**: Mentioned, but detailed layer/node specifications are NOT SPECIFIED IN PAPER.
- **Classical layers/activations**: NOT SPECIFIED IN PAPER.
- **Optimizer**: NOT SPECIFIED IN PAPER.

## 7. Quantum Neural Network
- **Number of qubits**: NOT SPECIFIED IN PAPER.
- **Quantum circuit layers/depth**: NOT SPECIFIED IN PAPER.
- **Input encoding strategy**: NOT SPECIFIED IN PAPER.
- **Quantum gates used**: NOT SPECIFIED IN PAPER.
- **Measurement/Output transformation**: NOT SPECIFIED IN PAPER.

## 8. Federated Learning
- **Aggregation logic**: Mentioned as part of QFRL, but exact aggregation rule (e.g., FedAvg) and frequency are NOT SPECIFIED IN PAPER.
- **Local vs Global training details**: NOT SPECIFIED IN PAPER.

## 9. Training Algorithm
- **Training procedure (Update rule)**: NOT SPECIFIED IN PAPER.
- **Exploration mechanism**: NOT SPECIFIED IN PAPER.
- **Target network / Soft update (Tau)**: NOT SPECIFIED IN PAPER.

## 10. Hyperparameters
- **Replay buffer capacity**: 6,000
- **Batch size**: 32
- **Episode length**: 5,000 steps
- **Discount factor (gamma)**: 0.95
- **Actor learning rate**: 0.001
- **Critic learning rate**: 0.0025
- **Training epochs**: 8,000

## 11. Explicitly Specified by Paper
- State dimension = 150
- Action dimension = 5
- The hyperparameters listed in Section 10
- The broad goal of optimizing Energy Efficiency via UAV, IRS, Power, and Offloading tuning.

## 12. Not Specified by Paper
- Exact 150-dimension state array mapping
- Exact 5-dimension action array mapping (and how 5 values control all VUEs/IRS)
- The step-by-step RL reward formula
- Quantum circuit design and qubit count
- Federated Learning synchronization schedule
- Many physical constants (e.g., beta_1, kappa, K_rician, bandwidth)

## 13. Current Implementation Assumptions
- We padded the 150-dim state with logically inferred properties.
- We mapped the 5-dim action to `[offload ratio, tx power, irs phase, uav x delta, uav y delta]`.
- We assumed reward equals exactly `completed_bits / total_energy` without temporal or penalty scalers.
- We use generic values for missing communication/computation constants (e.g., `beta = 2.2`, `kappa = 1e-28`).

---

# 14. IMPORTANT COMPARISON

| Component | Paper explicitly defines? | Current implementation | Safe for reproduction? |
|-----------|---------------------------|------------------------|------------------------|
| State dimension | YES (150) | 150 | YES |
| State features | NO | Guessed allocation | NO |
| Action dimension | YES (5) | 5 | YES |
| Action mapping | NO | Guessed 5D control | NO |
| Objective | YES (Energy Efficiency) | completed_bits / total_energy | YES (Conceptually) |
| Reward | NO | Exact ratio substitution | NO |
| Actor | NO | Placeholder | NO |
| Critic | NO | Placeholder | NO |
| QNN | NO | Not implemented | NO |
| FL aggregation | NO | Not implemented | NO |
| Training procedure | NO | Generic Loop Skeleton | NO |

---

# 15. TRAINING READINESS DECISION

TRAINING_READY = NO

The paper does not provide enough information in the currently available materials to implement the complete baseline without inventing core RL definitions. 

Specifically, the following mission-critical reproduction details are missing:
1. How 5 action dimensions map to the continuous control of potentially 30 vehicles, the UAV trajectory, and the IRS reflection matrix.
2. The exact construction of the 150-dimension state vector.
3. The exact mathematical reward formulation per step.
4. The Quantum Neural Network circuit architecture (qubits, gates, encoding, measurement).
5. The specific actor-critic update procedure and federated aggregation schedule.
