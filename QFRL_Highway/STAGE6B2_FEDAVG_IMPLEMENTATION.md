# STAGE 6B.2 FEDERATED AVERAGING (FedAvg) IMPLEMENTATION

## 1. Architecture
The QFRL FedAvg architecture utilizes a global `FederatedServer` and independent `FederatedClient` objects. The Server orchestrates a list of client states, performs sample-weighted aggregation, and broadcasts the updated models. Each Client acts as an isolated logical DDPG agent that runs local training within its own simulated environment instance.

## 2. Client Design
- Implemented as `FederatedClient` in `federated/client.py`.
- Each client wraps a local `DDPGAgent`, a `ReplayBuffer`, and a `QFRLEnvironment`.
- Enforces deterministic random seeds (`base_seed + client_id`) to ensure perfectly reproducible local experiences.
- Supports `get_parameters()`, `set_parameters()`, `collect_experience()`, and `local_train()`.

## 3. Server Design
- Implemented as `FederatedServer` in `federated/server.py`.
- Tracks the global parameter dictionary and handles aggregation logic.
- Manages saving/loading of `.pt` checkpoints, recording global configurations and agent state dictionaries without forcing the serialization of heavy client replay buffers.

## 4. FedAvg Equation
The system performs exact sample-weighted parameter aggregation across the selected clients:
$$\theta_{\text{global}} = \sum_{i} \frac{n_i}{N} \theta_i$$
where $n_i$ represents the number of local experience samples the client possesses, and $N$ is the total samples across participating clients.

## 5. Parameter Synchronization
- Upon initialization or aggregation, the server broadcasts global weights.
- Both the **Actor** and **Critic** networks are aggregated identically.
- Target networks in the clients (`target_actor`, `target_critic`) are also directly synced to match the newly aggregated global model perfectly, providing a unified starting point for the next federated round.

## 6. Local Training
Clients collect fresh transitions using `collect_experience()` and optimize their models using local batches through `local_train()`. The gradients strictly update the local actor and critic structures before parameter states are returned to the server.

## 7. Test Configuration
- Configuration overrides in `train_fedavg.py`: 3 clients, 2 local steps per round, 2 global rounds.
- This configuration ensures execution remains exceptionally fast (a smoke test).

## 8. Test Results
- All unit and mathematical tests in `tests/test_fedavg.py` **PASS**.
- The model shape alignment rules trigger a failure if clients mutate architecturally.
- The toy deterministic aggregation test matches its mathematical floating-point target perfectly.

## 9. Paper-Specified vs Implementation Choices
- **Implementation Choices**:
  - The use of exactly 30 logical clients simulated locally is an implementation approach, rather than true distributed deployment.
  - The FedAvg aggregation relies strictly on sample count weighting, a standard federated technique since exact formulas aren't explicitly outlined in the QFRL paper.
  - Target network hard-sync on every round is an implementation choice prioritizing stability.
  - Client seed offset `(base_seed + client_id)` ensures determinism across logical environments.

## 10. Known Limitations
- The underlying `TorchQuantum` circuits remain mocked on this Windows environment. This prevents actual quantum execution, though the macroscopic synchronization algorithms execute cleanly.
- Simulating 30 clients sequentially on a single CPU will scale poorly when moving to deep quantum circuits.

## 11. Next Stage
**STAGE 7: Kaggle/Linux REAL TorchQuantum Validation**
We will port this implementation to Kaggle (or a Linux environment), reinstate the real `TorchQuantum` library, and validate that quantum gradients successfully flow through the entire federated backpropagation tree.
