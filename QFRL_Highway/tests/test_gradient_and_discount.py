import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import torch
import torch.nn as nn
import yaml

from rl.classical_actor import ClassicalDDPGAgent, ClassicalActor, Critic
from rl.ddpg import DDPGAgent

def test_discounted_vs_undiscounted_energy_rank_inversion():
    """
    Demonstrates mathematically that discounted returns (gamma < 1) do NOT
    strictly preserve the ordering of undiscounted total energy.
    
    Sequence A: Front-loaded energy [60.0 J, 10.0 J]
      - Undiscounted total energy: 70.0 J
      - Aggregate EE (for constant 1 Mb/step): 2 Mb / 70 J = 0.02857 Mb/J
      - Discounted return (gamma=0.95): -60 - 0.95*10 = -69.50 J
      
    Sequence B: Back-loaded energy [10.0 J, 62.0 J]
      - Undiscounted total energy: 72.0 J (worse physical efficiency)
      - Aggregate EE: 2 Mb / 72 J = 0.02778 Mb/J
      - Discounted return (gamma=0.95): -10 - 0.95*62 = -68.90 J (higher return!)
      
    Result: Sequence B has WORSE total energy but HIGHER discounted return,
    proving that discounted DDPG objective is not strictly equivalent to
    undiscounted total energy minimization.
    """
    gamma = 0.95
    seq_A = np.array([60.0, 10.0])
    seq_B = np.array([10.0, 62.0])
    
    total_e_A = np.sum(seq_A) # 70.0
    total_e_B = np.sum(seq_B) # 72.0
    
    disc_ret_A = -seq_A[0] - gamma * seq_A[1] # -69.5
    disc_ret_B = -seq_B[0] - gamma * seq_B[1] # -68.9
    
    # Sequence A has LOWER total energy
    assert total_e_A < total_e_B, "Sequence A must have strictly lower total energy"
    
    # Sequence B has HIGHER discounted return (-68.9 > -69.5)
    assert disc_ret_B > disc_ret_A, "Sequence B must have strictly higher discounted return"
    
    # Demonstrating the rank inversion
    energy_diff = total_e_A - total_e_B # negative (< 0)
    return_diff = disc_ret_A - disc_ret_B # negative (< 0)
    # Higher return should correlate with lower energy (energy_diff * return_diff < 0)
    # Here both are negative, meaning the ordering is inverted!
    assert (energy_diff < 0 and return_diff < 0), "Rank inversion confirmed between discounted return and total energy"

def test_terminal_state_bootstrap_masking():
    """
    Verifies that target Q computation masks the bootstrap target
    when done == 1, and includes the bootstrap target when done == 0.
    """
    gamma = 0.95
    reward = torch.tensor([[-50.0], [-50.0]])
    next_q = torch.tensor([[100.0], [100.0]])
    
    # Transition 0 is not terminal (done=0); Transition 1 is terminal (done=1)
    done = torch.tensor([[0.0], [1.0]])
    
    target_q = reward + (1.0 - done) * gamma * next_q
    
    # For non-terminal: -50.0 + 1 * 0.95 * 100.0 = -50.0 + 95.0 = 45.0
    assert torch.isclose(target_q[0, 0], torch.tensor(45.0)), f"Expected 45.0, got {target_q[0, 0]}"
    
    # For terminal: -50.0 + 0 * 0.95 * 100.0 = -50.0
    assert torch.isclose(target_q[1, 0], torch.tensor(-50.0)), f"Expected -50.0, got {target_q[1, 0]}"

def test_actor_backward_pass_critic_isolation():
    """
    Verifies that during actor optimization:
    1. Critic parameters receive gradients during actor_loss.backward()
    2. actor_optimizer.step() updates ONLY actor parameters and leaves critic parameters untouched.
    """
    config = {
        'rl': {'state_dim': 150, 'action_dim': 5, 'gamma': 0.95},
        'device': 'cpu'
    }
    agent = ClassicalDDPGAgent(state_dim=150, action_dim=5, config=config)
    
    # Random batch
    batch_states = torch.randn(32, 150)
    
    # Zero gradients
    agent.actor.zero_grad()
    agent.critic.zero_grad()
    
    # Actor loss computation
    actor_loss = -agent.critic(batch_states, agent.actor(batch_states)).mean()
    actor_loss.backward()
    
    # 1. Critic parameters MUST receive non-zero gradients from the computational graph
    critic_grads = [p.grad for p in agent.critic.parameters() if p.grad is not None]
    assert len(critic_grads) > 0, "Critic parameters must receive gradients during actor_loss.backward()"
    critic_grad_norm = sum(p.grad.norm().item()**2 for p in agent.critic.parameters() if p.grad is not None)**0.5
    assert critic_grad_norm > 0.0, "Critic grad norm should be non-zero"
    
    # Record parameters before optimizer step
    actor_weights_before = [p.clone() for p in agent.actor.parameters()]
    critic_weights_before = [p.clone() for p in agent.critic.parameters()]
    
    # Step actor optimizer
    agent.actor_optimizer.step()
    
    # 2. Actor parameters should change (if not saturated)
    actor_delta = sum((p_after - p_bef).norm().item()**2 for p_after, p_bef in zip(agent.actor.parameters(), actor_weights_before))**0.5
    assert actor_delta > 0.0, "Actor optimizer must change actor parameters"
    
    # 3. Critic parameters MUST remain exactly identical (delta == 0.0)
    critic_delta = sum((p_after - p_bef).norm().item()**2 for p_after, p_bef in zip(agent.critic.parameters(), critic_weights_before))**0.5
    assert critic_delta == 0.0, f"Critic parameters must NOT be changed by actor_optimizer.step(), got delta={critic_delta}"

def test_classical_checkpoint_saturation_and_float32_underflow():
    """
    Verifies on an actual saved classical checkpoint (Seed 42, Ep 15):
    - Saturation fraction |a| > 0.95 is 100%
    - Saturation fraction |a| > 0.99 is 100%
    - Float32 actor gradient norm is exactly 0.0 due to Tanh derivative underflow.
    """
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "results", "checkpoints", "classical_seed42_ep15.pt")
    if not os.path.exists(ckpt_path):
        pytest.skip(f"Checkpoint not found at {ckpt_path}")
        
    ckpt = torch.load(ckpt_path, map_location='cpu')
    actor = ClassicalActor(state_dim=150, action_dim=5)
    actor.load_state_dict(ckpt['actor_state_dict'])
    actor.eval()
    
    critic = Critic(state_dim=150, action_dim=5)
    critic.load_state_dict(ckpt['critic_state_dict'])
    critic.eval()
    
    from environment.environment import QFRLEnvironment
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "paper_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    env = QFRLEnvironment()
    env.initialize_from_config(config)
    
    # Generate 10 reproducible valid environment states
    np.random.seed(999)
    env_states = []
    s = env.reset()
    env_states.append(s)
    for _ in range(9):
        s_next, _, done, _ = env.step(np.zeros(5))
        env_states.append(s_next)
        if done:
            s = env.reset()
        else:
            s = s_next
    states = torch.FloatTensor(np.array(env_states))
    
    with torch.no_grad():
        actions = actor(states)
        
    a = actions.numpy()
    sat_95 = np.mean(np.abs(a) > 0.95)
    sat_99 = np.mean(np.abs(a) > 0.99)
    
    assert sat_95 == 1.0, f"Expected 100% saturation at 0.95, got {sat_95}"
    assert sat_99 == 1.0, f"Expected 100% saturation at 0.99, got {sat_99}"
    
    # Backward pass
    actor.train()
    actor.zero_grad()
    critic.zero_grad()
    
    actor_loss = -critic(states, actor(states)).mean()
    actor_loss.backward()
    
    total_actor_grad_norm = sum(p.grad.norm().item()**2 for p in actor.parameters() if p.grad is not None)**0.5
    assert total_actor_grad_norm == 0.0, f"Expected exactly 0.0 float32 actor gradient norm, got {total_actor_grad_norm}"
