# STAGE 5 AUDIT

## 1. What the paper explicitly specifies
- **State Dimension:** 150
- **Action Dimension:** 5
- **Replay Buffer:** 6,000
- **Batch Size:** 32
- **Episode Length:** 5,000
- **Discount Factor:** 0.95
- **Actor LR:** 0.001
- **Critic LR:** 0.0025
- **Training Epochs:** 8,000
- **Objective:** Energy Efficiency = completed_bits / total_energy
- **Action components:** Mentions optimization variables related to task offloading, transmit power, IRS phase, and UAV trajectory.
- **Energy Equations:** E_local = kappa * C * f_v^2, E_UAV = kappa * C * f_u^2. 
- **Time Equations:** T_local = C / f_v, etc.
- **Communication Equations:** Channel path losses (G1, G2, G3), SNR, Shannon capacity.

## 2. What the paper does not explicitly specify
- **Exact State Feature Ordering:** The 150-dimension vector mapping is never explicitly laid out feature-by-feature.
- **Exact Action Feature Mapping:** The 5-dimension continuous vector is not explicitly broken down (e.g., whether offloading is one variable or per-vehicle, whether IRS phase is one scalar or vector).
- **Exact Normalization Ranges:** The physical boundaries for the Min-Max scaler are not listed explicitly.
- **Missing Parameters:** Many communication constants (`beta_1`, `beta_2`, `kappa`, `input_bits`, `bandwidth_hz`) are missing.

## 3. Current Implementation
- State mapping is implemented as an explicit layout accommodating the missing paper mapping.
- Action mapping is structured logically across the 5 dimensions.
- Environment orchestrates the imported Stage 3 & 4 modules to compute physical values natively.
- Energy efficiency directly measures completed bits / total energy. 

## 4. Implementation Assumptions
- **Action Assumption:** We mapped Action to `[offloading ratio, transmit power, irs_phase, uav_delta_x, uav_delta_y]`. This is an IMPLEMENTATION ASSUMPTION.
- **State Assumption:** 150 dimensions are padded using UAV, RSU, Vehicles and misc environment stats. This is an IMPLEMENTATION ASSUMPTION.
- **Task completion Assumption:** A task is considered completed if it finishes within the step. For now, we assume local tasks finish and offloaded tasks finish within `T = max(T_tx + T_uav, T_local)`.

## 5. Equations Verified
- Local computing time: `C / f_v`
- Local computing energy: `kappa * C * f_v^2`
- SNR: `S / (I + N)`
- Shannon Rate: `B * log2(1 + SNR)`
- Path Loss G1, G2, G3 models.

## 6. Equations requiring clarification
- Joint optimization of multiple vehicle offloading ratios using a single 5D action vector: The paper suggests a joint problem but mapping 30 vehicles to 1 dimension implies a uniform offloading policy or a clustered approach not fully defined.

## 7. Parameters Verified
- Evaluated parameters: `state_dim`, `action_dim`, `replay_buffer_size`, `batch_size`, `episode_length`, `gamma`, `actor_learning_rate`, `critic_learning_rate`, `training_epochs`. All match paper strictly.

## 8. Parameters missing from paper
- `beta_1`, `beta_2`, `beta_3`, `beta_4`
- `kappa`
- `K_rician`
- `wavelength_m`
- `irs_element_spacing_m`
- `bandwidth_hz`
- `input_bits`
- `cycles_per_bit`

## 9. Known limitations
- The continuous 5D action vector requires forcing uniform decisions across all 30 VUEs unless an attention or multi-agent formulation is clarified.
- NLoS fading vectors are randomly drawn without a defined temporal correlation model.

## 10. Whether the environment is ready for QFRL training
TRAINING_READY = NO
(We are updating the environment and test scripts to fully remove the unverified approximation in `step()` and properly trace communication -> rate -> computation -> energy).
