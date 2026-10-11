# Decisions

1. 150 → 4 classical compression is an implementation choice.
2. 4 quantum outputs are an implementation choice.
3. 4D Pauli-Z measurement is an implementation choice.
4. Exact Rx/Ry/Rz order is an implementation choice.
5. Ring CNOT topology is an implementation choice.
6. 5D action interpretation is an implementation assumption.
7. One common IRS phase across 50 elements is an implementation assumption.
8. completed_bits / total_energy is an implementation choice.
9. DDPG selected as the classical RL algorithm.
10. FedAvg will be implemented only after single-client validation.
11. Real TorchQuantum must eventually be tested on Kaggle/Linux.
12. Highway modifications will happen only after baseline reproduction.
13. Interpretation of paper's "8000 training epochs": Due to the prohibitive computational cost of 8000 episodes × 5000 steps (40,000,000 steps ≈ 9.2 GPU days on quantum simulator), baseline training experiments use a controlled multi-seed pilot and baseline configuration (50 steps/episode) with gradient update per step, documenting the mapping explicitly as an IMPLEMENTATION ASSUMPTION.
14. Reward scale and physical fidelity: Raw physical reward represents instantaneous energy efficiency (completed_bits / total_energy ≈ 10^5 bits/Joule), resulting in raw critic MSE losses on the order of 10^10. To maintain physical integrity, reward formulas are not arbitrarily modified or scaled down.
