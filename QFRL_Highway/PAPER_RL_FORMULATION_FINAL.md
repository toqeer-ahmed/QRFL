# PAPER RL FORMULATION (FINAL CORRECTED EXTRACTION)

This document extracts the exact mathematical and architectural definitions from the paper information established for this project. It strictly distinguishes between explicit paper facts, mathematical implications, and current implementation assumptions.

## 1. State
- **State dimension**: 150 (PAPER-SPECIFIED)
- **State variables/features**: The paper implies state variables representing the physical environment required for decision making, but the exact feature construction and ordering is NOT SPECIFIED.
- **Current Implementation**: We allocate the 150 dimensions to UAV position, RSU position, vehicle positions/CPU/task size, and padding. This is an **IMPLEMENTATION ASSUMPTION**, not an exact paper reproduction of the state vector.

## 2. Action Variables vs Action Dimension
- **Action dimension**: 5 (PAPER-SPECIFIED)
- **Optimization Variables**: The paper mathematically formulates optimization variables relating to:
  - Task offloading
  - Transmit power
  - IRS phase configuration
  - UAV trajectory
- **RL Action Mapping**: The fact that these optimization variables exist does NOT automatically prove they correspond one-to-one to the five RL action dimensions. The exact 5-dimensional action-vector mapping is NOT SPECIFIED.
- **Current Implementation**: 
  - `action[0]` = offloading ratio
  - `action[1]` = transmit power
  - `action[2]` = IRS phase
  - `action[3]` = UAV X movement
  - `action[4]` = UAV Y movement
  - This is an **IMPLEMENTATION ASSUMPTION** and is not the exact paper mapping.

## 3. Objective vs Reward
- **Optimization Objective**: Energy efficiency (completed task bits / total energy) is the explicit objective. (PAPER-SPECIFIED)
- **RL Reward**: Exact RL reward construction is NOT SPECIFIED. (The paper does not explicitly give a complete step-by-step reward equation).
- **Current Implementation**: `reward = completed_bits / total_energy`. This is an **IMPLEMENTATION CHOICE BASED ON THE PAPER'S ENERGY-EFFICIENCY OBJECTIVE**.

## 4. Quantum Neural Network — Paper-Specified Configuration
The paper explicitly defines the QNN configuration as follows:
- **Qubits**: 4 (PAPER-SPECIFIED)
- **Encoding**: Angle encoding (PAPER-SPECIFIED)
- **Initialization**: Hadamard (PAPER-SPECIFIED)
- **Rotation gates**: Rx, Ry, Rz (PAPER-SPECIFIED)
- **Entanglement**: CNOT (PAPER-SPECIFIED)
- **Circuit depth/layers**: 12 (PAPER-SPECIFIED)
- **Measurement shots**: 2048 (PAPER-SPECIFIED)
- **Framework**: TorchQuantum (PAPER-SPECIFIED)
- **Execution**: Classical simulation (PAPER-SPECIFIED)

Any other quantum architecture details not listed here remain unspecified.

## 5. Actor-Critic
- **Hyperparameters**: Actor LR = 0.001, Critic LR = 0.0025, Replay Buffer = 6000, Batch Size = 32, Discount Factor = 0.95, Training Epochs = 8000. (PAPER-SPECIFIED)
- **Architecture specifics**: Exact number of classical layers, nodes, and activation functions for the classical components are NOT SPECIFIED.
- **Current Implementation**: Any standard actor-critic mechanism built to bridge these components is an **IMPLEMENTATION CHOICE**.

## 6. Federated Learning
- **Details**: Local agents/clients, global aggregation, and synchronization are core to the QFRL formulation.
- **Mathematical Specification**: The exact aggregation logic (e.g., FedAvg) and frequency are NOT SPECIFIED.

## 7. Parameter Classification

| Parameter | Value | Classification | Evidence/Note |
|---|---:|---|---|
| State dimension | 150 | PAPER | |
| Action dimension | 5 | PAPER | |
| Vehicle count | 30 | PAPER | |
| VT max speed | 40 km/h | PAPER | |
| UAV max speed | 36 km/h | PAPER | |
| UAV altitude | 300 m | PAPER | |
| VT max TX power | 20 dBm | PAPER | |
| Noise power | -120 dBm | PAPER | |
| VT CPU | 4 GHz | PAPER | |
| RSU CPU | 9 GHz | PAPER | |
| UAV CPU | 12 GHz | PAPER | |
| IRS elements | 50 | PAPER | |
| Replay buffer | 6000 | PAPER | |
| Batch size | 32 | PAPER | |
| Episode length | 5000 | PAPER | |
| Gamma | 0.95 | PAPER | |
| Actor LR | 0.001 | PAPER | |
| Critic LR | 0.0025 | PAPER | |
| Training epochs | 8000 | PAPER | |
| Qubits | 4 | PAPER | |
| Quantum layers | 12 | PAPER | |
| Shots | 2048 | PAPER | |

## 8. Final Readiness Classification

Based on the audit of explicitly specified vs missing mathematical formulations, the readiness of the components is classified as follows:

**READY:**
- Physical environment
- Communication model
- Computation model
- QNN basic configuration
- Reported hyperparameters

**PARTIALLY SPECIFIED:**
- State construction
- Action mapping
- Exact reward construction
- Actor/critic architecture
- FL aggregation/training details

**NOT SPECIFIED:**
- Anything genuinely absent from the paper (e.g., exact step formulas tying the 5D action to all entities, exact FL sync frequency).

**BASELINE_REPRODUCTION_STATUS = PARTIALLY_SPECIFIED**

**Explanation:** While the physical simulation environment, equations, and exact hyperparameters (including QNN specs) are fully defined and implemented, the "glue" that binds them to standard Reinforcement Learning—specifically the exact mathematical mapping of the 150-dim state vector, the 5-dim action vector, and the reward calculation per time-step—is missing. Thus, any functional RL loop relies on our own architectural implementation assumptions.
