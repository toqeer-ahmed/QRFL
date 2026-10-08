# Stage 5: QFRL RL Formulation & Training Pipeline Fidelity Report

This document reports on the implementation of the RL framework for the QFRL reproduction project.

## 1. Paper-Specified State Dimension
**Reported Dimension:** 150
**Implemented:** Yes, strictly enforced to `(150,)` via zero padding where necessary.

## 2. Paper-Specified Action Dimension
**Reported Dimension:** 5
**Implemented:** Yes, the continuous action vector uses a length of `5` normalized between `[-1.0, 1.0]`.

## 3. Exact State Features Explicitly Supported by the Paper
The paper mentions tracking:
- Vehicle/task information (task bits, VUE CPU frequency)
- Communication/channel information
- Computation information
- UAV/IRS/Environment states (positions)

## 4. Features Whose Ordering/Dimensional Allocation is not Explicitly Specified
The exact mapping to the 150 dimensions was left unspecified in the paper. 

**Implementation Choice:**
To systematically populate the 150 dimensions, we constructed a fixed allocation strategy in `rl/state.py`:
- **UAV Position**: 2 dims (x, y)
- **RSU Position**: 2 dims (x, y)
- **Vehicles/Tasks**: 120 dims (4 dimensions each for up to 30 vehicles representing x, y, cpu_freq, and task input bits). 
- **Misc Environment**: 26 dims (UAV CPU, RSU CPU, IRS base positions, and padding elements to strictly reach the 150-dim requirement without breaking).
All components are bounded and normalized to `[0, 1]` using min-max scaling.

## 5. Exact Action Variables Explicitly Supported by the Paper
The optimization variables include:
- Task offloading
- Transmit power
- IRS phase configuration
- UAV trajectory

## 6. Any Action Mapping Assumptions
Because the continuous continuous multi-dimensional mapping of the 5 action parameters wasn't specified, we map them as follows in `rl/action.py`:
- `action[0]`: **Offloading Ratio** (scaled `[0.0, 1.0]`)
- `action[1]`: **Transmit Power** (scaled `[0, max_transmit_power_dbm]`)
- `action[2]`: **IRS Phase** (scaled to `[-pi, pi]`)
- `action[3]`: **UAV Delta X** (trajectory speed scaled to `[-max_uav_speed, max_uav_speed]`)
- `action[4]`: **UAV Delta Y** (trajectory speed scaled to `[-max_uav_speed, max_uav_speed]`)
Constraint clipping is enforced to prevent environment breakage (e.g., negative task bits, non-physical phase constraints).

## 7. Reward Definition
**Reported:** Energy Efficiency
**Implemented:** Yes, directly mapped as: `reward = completed_bits / total_energy`. It accurately reflects the physical values modeled. No artificial weights, latency, or reliability penalty modifiers were included.

## 8. Episode Definition
**Reported:** Episode length = 5,000 steps.
**Implemented:** Added `episode_length: 5000` to `configs/paper_config.yaml`. Managed by the `env.step()` termination signal.

## 9. Replay Buffer
**Reported:** 6,000 capacity.
**Implemented:** Yes, created `ReplayBuffer` in `rl/replay_buffer.py` that buffers `(s, a, r, s', d)` transitions up to size 6,000. Configured in `paper_config.yaml`.

## 10. Batch Size
**Reported:** 32.
**Implemented:** Configured in `paper_config.yaml`. Added default sampling of size 32 into `ReplayBuffer.sample()`.

## 11. Discount Factor
**Reported:** 0.95.
**Implemented:** Configured in `paper_config.yaml`.

## 12. Learning Rates
**Reported:** Actor LR = 0.001, Critic LR = 0.0025.
**Implemented:** Configured in `paper_config.yaml`.

## 13. Training Epochs
**Reported:** 8,000.
**Implemented:** Configured in `paper_config.yaml`. (Actual large-scale loops are deferred to Kaggle execution).

## 14. Any Implementation Assumptions
- **Placeholder Testing:** We assume standard Gaussian/Uniform generation for the placeholder agent testing.
- **Kaggle Run Setup:** Pathing has been made relative inside `experiments/kaggle_train.py` for cloud flexibility. 
- **PyTorch / Quantum Modules Delayed:** The integration of the QNN policy was explicitly avoided at this stage to prevent local bloat, and `torch` dependencies are mocked or avoided for lightweight environment setup.

## 15. Action Definition: Paper Evidence vs Implementation Assumption
| Dimension | Physical Variable | Paper explicitly specifies ordering? | Current Implementation | Status |
|-----------|-------------------|--------------------------------------|------------------------|--------|
| `action[0]` | Task offloading ratio | NO | `action[0]` maps to ratio (0 to 1) | IMPLEMENTATION ASSUMPTION |
| `action[1]` | Transmit Power | NO | `action[1]` maps to `[0, max_power]` | IMPLEMENTATION ASSUMPTION |
| `action[2]` | IRS Phase | NO | `action[2]` maps to `[-pi, pi]` | IMPLEMENTATION ASSUMPTION |
| `action[3]` | UAV Delta X | NO | `action[3]` maps to UAV X velocity | IMPLEMENTATION ASSUMPTION |
| `action[4]` | UAV Delta Y | NO | `action[4]` maps to UAV Y velocity | IMPLEMENTATION ASSUMPTION |

## 16. Paper Parameters vs Implementation Assumptions
| Parameter | Value | Source | Status |
|-----------|-------|--------|--------|
| `state_dim` | 150 | Paper | [PAPER] |
| `action_dim` | 5 | Paper | [PAPER] |
| `replay_buffer_size` | 6000 | Paper | [PAPER] |
| `batch_size` | 32 | Paper | [PAPER] |
| `episode_length` | 5000 | Paper | [PAPER] |
| `gamma` | 0.95 | Paper | [PAPER] |
| `actor_learning_rate` | 0.001 | Paper | [PAPER] |
| `critic_learning_rate` | 0.0025 | Paper | [PAPER] |
| `training_epochs` | 8000 | Paper | [PAPER] |
| `beta_1, beta_2, beta_3, beta_4` | Configurable | Not explicitly specified | [UNSPECIFIED_IN_PAPER] |
| `kappa` | 1e-28 | Generic approximation | [IMPLEMENTATION_ASSUMPTION] |
| `K_rician` | 10.0 | Typical value | [UNSPECIFIED_IN_PAPER] |
| `noise_power_dbm` | -120 | Paper | [PAPER] |

## 17. State Definition
| State feature | Dimension(s) | Physical meaning | Paper explicitly specified? | Implementation assumption? |
|---------------|--------------|------------------|-----------------------------|----------------------------|
| UAV Pos | 2 | UAV X, Y | NO | YES |
| RSU Pos | 2 | RSU X, Y | NO | YES |
| Vehicle Info | 120 (4x30) | VUE X, Y, CPU, Task Size | NO | YES |
| Misc / Pad | 26 | IRS Pos, UAV CPU, pad zeros | NO | YES |

