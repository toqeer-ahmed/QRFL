import os
import sys
import tempfile
import pytest
import torch
import numpy as np

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.classical_actor import ClassicalActor, ClassicalDDPGAgent
from rl.replay_buffer import ReplayBuffer

def test_classical_actor_dimensions_and_bounds():
    actor = ClassicalActor(state_dim=150, action_dim=5, hidden_dims=(64, 64))
    
    # Test batch 1
    x1 = torch.randn(1, 150)
    out1 = actor(x1)
    assert out1.shape == (1, 5)
    assert torch.all(out1 >= -1.0) and torch.all(out1 <= 1.0)
    
    # Test batch 32
    x32 = torch.randn(32, 150)
    out32 = actor(x32)
    assert out32.shape == (32, 5)
    assert torch.all(out32 >= -1.0) and torch.all(out32 <= 1.0)

def test_classical_actor_parameter_count():
    actor = ClassicalActor(state_dim=150, action_dim=5, hidden_dims=(64, 64))
    param_count = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    expected_count = (150 * 64 + 64) + (64 * 64 + 64) + (64 * 5 + 5)
    assert param_count == expected_count == 14149

def test_classical_ddpg_agent_update():
    agent = ClassicalDDPGAgent(state_dim=150, action_dim=5)
    buffer = ReplayBuffer(state_dim=150, action_dim=5, max_size=100)
    
    # Populate buffer
    for _ in range(35):
        s = np.random.randn(150).astype(np.float32)
        a = np.random.uniform(-1, 1, 5).astype(np.float32)
        r = float(np.random.randn())
        s_next = np.random.randn(150).astype(np.float32)
        d = False
        buffer.add(s, a, r, s_next, d)
        
    al, cl = agent.train(buffer, batch_size=32)
    assert al is not None and cl is not None
    assert np.isfinite(al) and np.isfinite(cl)

def test_classical_ddpg_checkpoint_bit_exact():
    agent1 = ClassicalDDPGAgent(state_dim=150, action_dim=5)
    agent2 = ClassicalDDPGAgent(state_dim=150, action_dim=5)
    
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        agent1.save_checkpoint(tmp_path, episode=5, seed=42)
        ep, seed, _ = agent2.load_checkpoint(tmp_path)
        assert ep == 5
        assert seed == 42
        
        # Verify exact weight match
        for p1, p2 in zip(agent1.actor.parameters(), agent2.actor.parameters()):
            assert torch.allclose(p1, p2)
        for p1, p2 in zip(agent1.critic.parameters(), agent2.critic.parameters()):
            assert torch.allclose(p1, p2)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
