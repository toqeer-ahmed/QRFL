# STAGE 6A: QFRL ARCHITECTURE & IMPLEMENTATION DESIGN

This document defines the architectural pipeline for the QFRL reproduction project. It connects the 150-dimensional physical state, the 4-qubit Quantum Neural Network, the 5-dimensional action space, and the overarching Federated Learning loop.

## 1. PAPER FACTS
The following elements are firmly established by the paper:
- **State dimension**: 150
- **Action dimension**: 5
- **QNN Qubits**: 4
- **QNN Encoding**: Angle encoding
- **QNN Initialization**: Hadamard initialization
- **QNN Gates**: Rx, Ry, Rz rotation gates
- **QNN Entanglement**: CNOT gates
- **QNN Depth**: 12 quantum circuit layers
- **QNN Execution**: 2048 measurement shots, TorchQuantum framework, classical simulation
- **Replay buffer**: 6000
- **Batch size**: 32
- **Discount factor (gamma)**: 0.95
- **Actor LR**: 0.001
- **Critic LR**: 0.0025
- **Training epochs**: 8000
- **Optimization Objective**: Energy efficiency

**Note**: The paper does NOT explicitly specify the complete state feature ordering, the 5D action mapping, the complete step reward equation, the classical actor/critic architecture linking these components, or the complete FL aggregation mechanism.

## 2. STATE PIPELINE
**Data Flow**: 
150-dimensional state $\rightarrow$ Classical Dimensionality Reduction $\rightarrow$ Quantum Angle Encoding

**Implementation Design**:
To encode 150 features into a 4-qubit quantum circuit using standard angle encoding, the dimensionality must be reduced to 4. 
- **Mechanism**: A classical linear layer (or small MLP) transforming `R^150 -> R^4` prior to the quantum encoding.
- **Why?**: Angle encoding directly embeds scalar values into qubit rotation angles (e.g., via $Ry$ or $Rx$). A 4-qubit system can natively accept 4 angles per layer.
- **Classification**: **IMPLEMENTATION ASSUMPTION** (The exact compression strategy from 150 to 4 was omitted by the authors).

## 3. QUANTUM ACTOR
**Architecture**: Hybrid Classical-Quantum Actor Network
1. **Classical Input Layer**: $R^{150} \rightarrow R^4$
2. **Quantum Circuit (TorchQuantum)**:
   - **Initialization**: Hadamard gates applied to all 4 qubits.
   - **Encoding**: The 4 classical outputs act as angles for $Rx/Ry/Rz$ gates.
   - **Variational Layers**: 12 trainable layers consisting of $Rx, Ry, Rz$ rotations and CNOT entanglement structures.
   - **Measurement**: Expectation values (Pauli-Z) measured over 2048 simulated shots.
3. **Classical Output Layer**: $R^4 \rightarrow R^5$
   - Translates the 4 expectation values into the 5 continuous dimensions required for the action.

**Classification**: The basic QNN structure is **PAPER-SPECIFIED**, but the classical linear wrapping to match the input/output boundaries is an **IMPLEMENTATION ASSUMPTION**.

## 4. 5-D ACTION OUTPUT
**Transformation Mapping**:
Raw Actor Output $\rightarrow$ Tanh activation (Bounded $[-1, 1]$) $\rightarrow$ Physical Scaling

| Action Index | RL Variable | Range (Normalized) | Physical Range | Bounding Method | IMPLEMENTATION ASSUMPTION |
|--------------|-------------|--------------------|----------------|-----------------|---------------------------|
| `action[0]` | Offloading Ratio | `[-1.0, 1.0]` | `[0.0, 1.0]` | Linear shift/scale | YES |
| `action[1]` | Transmit Power | `[-1.0, 1.0]` | `[0, 20] dBm` | Linear shift/scale | YES |
| `action[2]` | IRS Phase | `[-1.0, 1.0]` | `[-pi, pi]` | Scalar multiplication | YES |
| `action[3]` | UAV $\Delta x$ | `[-1.0, 1.0]` | `[-10, 10] m/s` | Scalar multiplication | YES |
| `action[4]` | UAV $\Delta y$ | `[-1.0, 1.0]` | `[-10, 10] m/s` | Scalar multiplication | YES |

## 5. CRITICAL IRS QUESTION
**The Problem**: The paper simulates 50 independent IRS elements, but the continuous action vector has only 5 total dimensions. Allocating one action dimension to the IRS makes independent phase control impossible.
**Proposed Solution**: `action[2]` acts as a single common phase-shift scalar uniformly applied to all 50 IRS elements. While this limits complex beamforming patterns, it is the only mathematically viable solution without expanding the paper-specified 5-dimension action space.
**Classification**: **IMPLEMENTATION ASSUMPTION**

## 6. ACTOR-CRITIC
To train continuous actions, an off-policy deterministic actor-critic algorithm is required. We select **Deep Deterministic Policy Gradient (DDPG)** (or Twin Delayed DDPG - TD3).
- **Actor Network**: The hybrid QNN described in Section 3.
- **Critic Network**: A classical MLP mapping $(State_{150}, Action_{5}) \rightarrow Q-Value_{1}$.
- **Target Networks**: EMA-updated (Soft update) target networks for stability.
- **Losses**: Standard TD-error for Critic, deterministic policy gradient for Actor.
**Classification**: **IMPLEMENTATION CHOICE** (The exact RL wrapper topology around the QNN was not specified).

## 7. REWARD
**Definition**: $r_t = \frac{\text{Completed Task Bits}_t}{\text{Total Energy Consumed}_t}$
- **Zero Energy Guard**: If total energy $\le 0$, it is clamped to $10^{-6}$ to prevent division by zero.
- **Task Completion Guard**: Tasks whose transmission time exceeds 10.0 seconds are considered dropped ($bits = 0$).
- **Scaling**: A scaling factor might be applied during training to keep rewards within standard bounds.
**Classification**: **IMPLEMENTATION CHOICE BASED ON PAPER OBJECTIVE**

## 8. FEDERATED LEARNING
**Minimum Viable Federated Loop**:
- **Clients**: Multiple simulated VUE groups or regional instances (assuming 1 client instance for baseline).
- **Local Training**: Clients perform DDPG updates (actor/critic loss gradients) locally using their replay buffers.
- **Global Aggregation**: Periodically (e.g., every $E$ episodes), the classical weights and variational quantum parameters are collected.
- **Method**: Federated Averaging (**FedAvg**).
- **Synchronization**: The averaged global model is broadcast back to local clients.
**Classification**: **IMPLEMENTATION CHOICE**

## 9. COMPLETE DATA FLOW
```text
      Global Server (Federated Averaging)
           |                ^
       Broadcast          Aggregate
           v                |
+-----------------------------------------------+
|                 Local Client                  |
|                                               |
|  Environment -> 150-D State (s_t)             |
|                       |                       |
|           +-----------v-----------+           |
|           | Classical Compression | R^150->R^4|
|           +-----------+-----------+           |
|                       |                       |
|           +-----------v-----------+           |
|           |   4-Qubit QNN Actor   |           |
|           +-----------+-----------+           |
|                       |                       |
|           +-----------v-----------+           |
|           | Classical Expansion   | R^4->R^5  |
|           +-----------+-----------+           |
|                       |                       |
|                  5-D Action (a_t)             |
|                       |                       |
|               Action Scaling                  |
|                       |                       |
|  Environment Step (Comm, Comp, Energy)        |
|                       |                       |
|  Next State (s_t+1), Reward (r_t), Done       |
|                       |                       |
|             Replay Buffer (6000)              |
|                       |                       |
|     +-----------------v-----------------+     |
|     | Critic Update (TD Error)          |     |
|     | Actor Update (Policy Gradient)    |     |
|     +-----------------------------------+     |
+-----------------------------------------------+
```

## 10. MATHEMATICAL DEFINITIONS
**State**: 
$s_t \in \mathbb{R}^{150}$

**Action**: 
$a_t \in \mathbb{R}^5, \quad a_t = \tanh(QNN(s_t))$

**Normalized Action Transformation**:
$\text{Physical Action} = \text{Scale}(a_t, bounds)$ (IMPLEMENTATION FORMULATION)

**Reward**:
$r_t = \frac{\sum B_{\text{completed}}}{\max(10^{-6}, \sum E_{\text{consumed}})}$ (IMPLEMENTATION FORMULATION)

**Actor Objective (Deterministic Policy Gradient)**:
$J(\theta_\mu) = \mathbb{E}_{s \sim \mathcal{D}} \left[ Q(s, \mu(s | \theta_\mu) | \theta_Q) \right]$ (IMPLEMENTATION FORMULATION)

**Critic Loss**:
$L(\theta_Q) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( Q(s, a | \theta_Q) - y \right)^2 \right]$
$y = r + \gamma (1 - d) Q'(s', \mu'(s' | \theta_{\mu'}) | \theta_{Q'})$ (IMPLEMENTATION FORMULATION)

**Federated Aggregation (FedAvg)**:
$W_{global} = \frac{1}{K} \sum_{k=1}^K W_{local}^k$ (IMPLEMENTATION FORMULATION)

## 11. PARAMETER CLASSIFICATION

| Component | Decision | Classification |
|---|---|---|
| State dimension | 150 | PAPER |
| Action dimension | 5 | PAPER |
| Qubits | 4 | PAPER |
| Quantum layers | 12 | PAPER |
| Shots | 2048 | PAPER |
| State feature ordering | Proposed | IMPLEMENTATION ASSUMPTION |
| State $\rightarrow$ QNN compression | Proposed (Linear R^150->R^4) | IMPLEMENTATION ASSUMPTION |
| Action mapping | Proposed (Offload, Tx Power, IRS, UAV dx, UAV dy) | IMPLEMENTATION ASSUMPTION |
| Action expansion | Proposed (Linear R^4->R^5) | IMPLEMENTATION ASSUMPTION |
| IRS phase representation | Proposed (1 common phase for 50 elements) | IMPLEMENTATION ASSUMPTION |
| Actor architecture | Proposed (Hybrid Classical-Quantum DDPG) | IMPLEMENTATION CHOICE |
| Critic architecture | Proposed (Classical MLP) | IMPLEMENTATION CHOICE |
| RL algorithm | Proposed (DDPG) | IMPLEMENTATION CHOICE |
| Reward formula | completed_bits / energy | IMPLEMENTATION CHOICE |
| FL aggregation | Proposed (FedAvg) | IMPLEMENTATION CHOICE |

## 12. REPRODUCIBILITY WARNING

**This implementation is a reconstruction of the QFRL framework based on the paper's reported system model, hyperparameters, and quantum configuration. The paper does not provide sufficient detail to reproduce every RL architectural decision exactly. Therefore, the missing RL/FL glue is explicitly implemented as engineering choices and must not be presented as directly reproduced paper architecture.**

## 13. IMPLEMENTATION PLAN

### Stage 6B
The next stage will implement the architecture defined in this document. 
This includes writing the TorchQuantum circuit, classical compression layers, the DDPG Critic, target network updates, and orchestrating the FedAvg synchronization loop.

(Stage 6B must NOT start until this document is internally consistent and approved.)
