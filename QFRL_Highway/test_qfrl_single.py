import torch
import numpy as np
import pytest
import os
import yaml

from quantum.qnn import QuantumActor
from rl.action_mapper import ActionMapper
from environment.irs_controller import IRSController
from rl.ddpg import Critic, DDPGAgent
from rl.replay_buffer import ReplayBuffer
from environment.environment import QFRLEnvironment

def get_config():
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'paper_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_1_actor_shape():
    actor = QuantumActor(state_dim=150, action_dim=5, n_qubits=4, n_layers=12, n_shots=2048)
    state = torch.randn(32, 150)
    action = actor(state)
    assert action.shape == (32, 5), f"Expected (32, 5), got {action.shape}"

def test_2_actor_bounds():
    actor = QuantumActor()
    state = torch.randn(32, 150)
    action = actor(state)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0), "Actor outputs not bounded in [-1, 1]"

def test_3_action_mapper():
    mapper = ActionMapper()
    raw_action = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    phys_action = mapper.decode_action(raw_action)
    
    assert phys_action['offload_ratio'] == 1.0
    assert phys_action['transmit_power'] == 20.0
    assert np.isclose(phys_action['irs_phase'], np.pi)
    # v_x and v_y for 1,1 before norm was 10,10. Norm is 14.14. After projection, each is 10/sqrt(2) = 7.07
    assert np.isclose(phys_action['uav_delta_x'], 7.0710678)
    assert np.isclose(phys_action['uav_delta_y'], 7.0710678)

def test_4_uav_velocity_constraint():
    mapper = ActionMapper()
    raw_action = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    phys = mapper.decode_action(raw_action)
    vx = phys['uav_delta_x']
    vy = phys['uav_delta_y']
    v_norm = np.sqrt(vx**2 + vy**2)
    assert v_norm <= 10.0 + 1e-5, f"Velocity norm {v_norm} exceeds 10 m/s"

def test_5_6_irs_controller():
    controller = IRSController(num_elements=50)
    phases = controller.generate_phase_vector(0.5)
    assert len(phases) == 50, f"Expected 50 phases, got {len(phases)}"
    assert all(p == 0.5 for p in phases), "Phases are not identical (common phase assumption violated)"

def test_7_critic_shape():
    critic = Critic(state_dim=150, action_dim=5)
    state = torch.randn(32, 150)
    action = torch.randn(32, 5)
    q = critic(state, action)
    assert q.shape == (32, 1), f"Expected (32, 1), got {q.shape}"

def test_8_replay_buffer():
    buf = ReplayBuffer(state_dim=150, action_dim=5, max_size=100)
    buf.add(np.zeros(150), np.zeros(5), 1.0, np.ones(150), False)
    assert len(buf) == 1
    
    # Fill to get a batch
    for _ in range(40):
        buf.add(np.zeros(150), np.zeros(5), 1.0, np.ones(150), False)
    
    batch = buf.sample(32)
    assert batch['states'].shape == (32, 150)
    assert batch['actions'].shape == (32, 5)
    assert batch['rewards'].shape == (32, 1)

def test_9_ddpg_update():
    agent = DDPGAgent(config={'gamma': 0.95})
    buf = ReplayBuffer(150, 5)
    for _ in range(40):
        buf.add(np.random.randn(150), np.random.randn(5), 1.0, np.random.randn(150), False)
        
    actor_loss, critic_loss = agent.train(buf, batch_size=32)
    assert actor_loss is not None
    assert not np.isnan(actor_loss) and not np.isinf(actor_loss)
    assert not np.isnan(critic_loss) and not np.isinf(critic_loss)

def test_10_env_transition():
    config = get_config()
    env = QFRLEnvironment()
    env.initialize_from_config(config)
    state = env.reset()
    
    agent = DDPGAgent(config=config)
    action = agent.select_action(state, add_noise=False)
    
    next_state, reward, done, info = env.step(action)
    
    assert next_state.shape == (150,)
    assert isinstance(reward, float)
    
    buf = ReplayBuffer(150, 5)
    buf.add(state, action, reward, next_state, done)
    assert len(buf) == 1

def test_11_short_training_run():
    config = get_config()
    env = QFRLEnvironment()
    env.initialize_from_config(config)
    
    agent = DDPGAgent(config=config)
    buf = ReplayBuffer(150, 5, max_size=1000)
    
    episodes = 2
    max_steps = 10
    
    for ep in range(episodes):
        state = env.reset()
        for step in range(max_steps):
            action = agent.select_action(state, add_noise=True)
            next_state, reward, done, info = env.step(action)
            buf.add(state, action, reward, next_state, done)
            
            if len(buf) > 32:
                agent.train(buf, batch_size=32)
                
            state = next_state
            if done:
                break

if __name__ == "__main__":
    test_1_actor_shape()
    test_2_actor_bounds()
    test_3_action_mapper()
    test_4_uav_velocity_constraint()
    test_5_6_irs_controller()
    test_7_critic_shape()
    test_8_replay_buffer()
    test_9_ddpg_update()
    test_10_env_transition()
    test_11_short_training_run()
    print("All tests passed!")
