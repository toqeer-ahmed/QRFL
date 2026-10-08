# STAGE 6A: QFRL ARCHITECTURE & IMPLEMENTATION DESIGN (REVISED)

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
A 4-qubit circuit does not universally require the original 150-dimensional state itself to be reduced to exactly four values. However, our chosen first implementation will use:
- **Mechanism**: A classical linear layer transforming `Linear(150, 4)` prior to the quantum encoding.
- **Why?**: It is the simplest low-assumption projection into the 4-angle capacity natively provided by a 4-qubit circuit.
- **Classification**: **IMPLEMENTATION CHOICE** (The exact compression strategy from 150 features to the QNN input was omitted by the authors).

## 3. QUANTUM ACTOR
**Architecture**: The proposed implementation is a hybrid quantum-classical actor-critic architecture in which the QNN is embedded inside the actor while the critic remains classical.

1. **Classical Input Layer**: $R^{150} \rightarrow R^4$ (IMPLEMENTATION CHOICE)
2. **Quantum Circuit (TorchQuantum)**:
   - **PAPER-SPECIFIED**: 4 qubits, 2048 shots, Hadamard initialization, Angle encoding, Rx/Ry/Rz and CNOT gates, 12 layers.
   - **IMPLEMENTATION CHOICE (Measurement Interface)**: Measuring one Pauli-Z expectation per qubit to produce a 4-dimensional quantum representation output.
3. **Classical Output Layer**: $R^4 \rightarrow R^5$ (IMPLEMENTATION CHOICE)
   - Translates the 4 quantum expectation values into the 5 continuous dimensions required for the action.

## 4. 5-D ACTION OUTPUT
**Transformation Mapping**:
Raw Actor Output $\rightarrow$ Tanh activation (Bounded $[-1, 1]$) $\rightarrow$ Physical Scaling

| Action Index | RL Variable | Range (Normalized) | Physical Range | Bounding Method | IMPLEMENTATION ASSUMPTION |
|--------------|-------------|--------------------|----------------|-----------------|---------------------------|
| `action[0]` | Offloading Ratio | `[-1.0, 1.0]` | `[0.0, 1.0]` | Linear shift/scale | YES |
| `action[1]` | Transmit Power | `[-1.0, 1.0]` | `[0, 20] dBm` | Linear shift/scale | YES |
| `action[2]` | IRS Phase | `[-1.0, 1.0]` | `[-pi, pi]` | Scalar multiplication | YES |
| `action[3]` | UAV $\Delta x$ | `[-1.0, 1.0]` | Velocity $v_x$ | See constraint logic | YES |
| `action[4]` | UAV $\Delta y$ | `[-1.0, 1.0]` | Velocity $v_y$ | See constraint logic | YES |

### UAV Movement Constraint
The paper specifies a UAV maximum speed of 36 km/h = 10 m/s. Treating independent $dx$ and $dy$ bounds in `[-10, 10]` could yield a resultant speed up to 14.14 m/s. 
- **Constraint**: $\sqrt{v_x^2 + v_y^2} \le 10$ m/s.
- **Mechanism**: After extracting the raw vector from the actor output, if the $L2$-norm exceeds 10 m/s, it is projected onto the radius-10 disk.
- **Classification**: **IMPLEMENTATION FORMULATION**

### Multi-Vehicle Ambiguity
**MAJOR PAPER-REPRODUCTION AMBIGUITY**: The paper does not clearly explain how a 5-dimensional action controls the decision variables of up to 30 VUEs. Therefore, we must NOT claim that this provides independent per-VUE control. The current hypothesis assumes 5 unified scalar variables applied across the environment, but interpretations will remain isolated in code so alternative architectures can be explored.

## 5. CRITICAL IRS QUESTION
**The Problem**: A 5-dimensional action cannot independently specify 50 arbitrary continuous IRS phase values. Therefore, a structured or parameterized representation is required if all 50 IRS elements are to be controlled through the 5-dimensional action. The paper does not provide enough detail to determine that representation.

**Possible Approaches**:
A. Common phase: $\phi_p = \phi$ for all $p$
B. Parameterized phase pattern: $\phi_p = f(p, action[2], ...)$
C. Other structured mapping

**Decision**: For the first implementation, common-phase control is a temporary baseline assumption only and must be isolated in code so it can be replaced without changing the actor.
**Classification**: **IMPLEMENTATION ASSUMPTION**

## 6. ACTOR-CRITIC
To train continuous actions, an off-policy deterministic actor-critic algorithm is required. 
- **Algorithm**: Deep Deterministic Policy Gradient (DDPG).
- **Reason**: The paper specifies a continuous action space, replay buffer, actor/critic learning rates, gamma, and batch size. DDPG provides a simple deterministic actor-critic baseline satisfying all these requirements with fewer additional hyperparameters than other algorithms.
- **Critic Network**: Classical MLP mapping $(State_{150}, Action_{5}) \rightarrow Q-Value_{1}$.
**Classification**: **IMPLEMENTATION CHOICE** (The paper does not explicitly specify DDPG).

## 7. REWARD
**Objective**: Energy Efficiency

**Reward Formulation**: $r_t = \frac{\text{Completed Task Bits}_t}{\text{Total Energy Consumed}_t}$
- **Definition**: The reward calculates completed bits and energy derived strictly from task workload, transmission time, computation time, simulation horizon, and actual environment constraints (without artificially hardcoded timeout penalties unless introduced later as configurable experimental parameters). 
- **Zero Energy Guard**: If energy $\le 0$, it is clamped to $10^{-6}$ for numerical stability.
**Classification**: **IMPLEMENTATION CHOICE BASED ON PAPER OBJECTIVE** (The step-by-step reward equation was not explicitly provided by the paper).

## 8. FEDERATED LEARNING
**Federated Averaging (FedAvg)**
- **Structure**: Global model $\rightarrow$ distribute to clients $\rightarrow$ local RL updates (DDPG) $\rightarrow$ collect model parameters $\rightarrow$ aggregate $\rightarrow$ broadcast global model.
- **Classification**: **IMPLEMENTATION CHOICE**

**Implementation Plan**:
- **STAGE 6B.1**: Single-client QFRL/DDPG (Allows the quantum actor, critic, reward, and replay pipeline to be validated before federation overhead).
- **STAGE 6B.2**: Multi-client federated extension.

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
$a_t \in \mathbb{R}^5, \quad a_t = \text{Scale}(\tanh(\text{Linear}_{4\rightarrow5}(\text{Measure}_{Pauli-Z}(\text{QNN}(\text{Linear}_{150\rightarrow4}(s_t))))))$ (IMPLEMENTATION FORMULATION)

**UAV Constraint Projection**:
If $\sqrt{v_x^2 + v_y^2} > 10$: $v_x, v_y = \frac{v_x, v_y}{\sqrt{v_x^2 + v_y^2}} \times 10$ (IMPLEMENTATION FORMULATION)

**Reward**:
$r_t = \frac{\sum B_{\text{completed}}}{\max(10^{-6}, \sum E_{\text{consumed}})}$ (IMPLEMENTATION FORMULATION)

**Actor Objective (Deterministic Policy Gradient)**:
$J(\theta_\mu) = \mathbb{E}_{s \sim \mathcal{D}} \left[ Q(s, \mu(s | \theta_\mu) | \theta_Q) \right]$ (IMPLEMENTATION FORMULATION)

**Critic Loss**:
$L(\theta_Q) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( Q(s, a | \theta_Q) - y \right)^2 \right]$
$y = r + \gamma (1 - d) Q'(s', \mu'(s' | \theta_{\mu'}) | \theta_{Q'})$ (IMPLEMENTATION FORMULATION)

**Federated Aggregation (FedAvg)**:
$W_{global} = \frac{1}{K} \sum_{k=1}^K W_{local}^k$ (IMPLEMENTATION FORMULATION)

## 11. REPRODUCIBILITY TABLE

| Component | Status |
|---|---|
| Physical environment | PAPER/IMPLEMENTED |
| Communication | PAPER/IMPLEMENTED |
| Computation | PAPER/IMPLEMENTED |
| State dimension | PAPER |
| Action dimension | PAPER |
| QNN qubits | PAPER |
| QNN gates | PAPER |
| QNN layers | PAPER |
| Measurement shots | PAPER |
| State compression | IMPLEMENTATION CHOICE |
| Quantum measurement output | IMPLEMENTATION CHOICE |
| Action mapping | IMPLEMENTATION ASSUMPTION |
| IRS representation | IMPLEMENTATION ASSUMPTION |
| DDPG | IMPLEMENTATION CHOICE |
| Reward | IMPLEMENTATION CHOICE |
| Critic architecture | IMPLEMENTATION CHOICE |
| FedAvg | IMPLEMENTATION CHOICE |
| FL synchronization | IMPLEMENTATION CHOICE |

## 12. REPRODUCIBILITY WARNING

**This implementation is a reconstruction of the QFRL framework based on the paper's reported system model, hyperparameters, and quantum configuration. The paper does not provide sufficient detail to reproduce every RL architectural decision exactly. Therefore, the missing RL/FL glue is explicitly implemented as engineering choices and must not be presented as directly reproduced paper architecture.**

## 13. IMPLEMENTATION PLAN

### Stage 6B
The next stage will implement the architecture defined in this document. 

Stage 6B must implement the assumptions as modular components so that each assumption can later be replaced without redesigning the entire system.

**STAGE_6A_STATUS = APPROVED_FOR_IMPLEMENTATION_WITH_EXPLICIT_ASSUMPTIONS**
