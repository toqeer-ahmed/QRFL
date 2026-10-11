import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from quantum.qnn import QuantumActor
from rl.ddpg import DDPGAgent
from rl.replay_buffer import ReplayBuffer
import numpy as np

def test_qnn_gradients():
    print("==================================================")
    print("QNN GRADIENT FLOW TEST")
    print("==================================================")
    
    actor = QuantumActor(state_dim=150, action_dim=5)
    
    # Check if we are using the mock
    try:
        import torchquantum as tq
    except ImportError:
        import mock_torchquantum as tq
    print(f"TorchQuantum version loaded: {getattr(tq, '__version__', 'UNKNOWN')}")
    
    # 1. State Input
    state = torch.randn(2, 150, requires_grad=True)
    
    # 2. Forward pass
    output = actor(state)
    print(f"Output shape: {output.shape}")
    print(f"Output finite: {torch.isfinite(output).all().item()}")
    
    # 3. Create a scalar loss
    loss = output.sum()
    loss.backward()
    
    # 4. Check gradients
    print("\n--- GRADIENT CHECK ---")
    
    # Check Linear(150,4)
    comp_grad = actor.state_compression[0].weight.grad
    if comp_grad is not None:
        print(f"Linear(150,4) [state_compression] grad exists: YES | finite: {torch.isfinite(comp_grad).all().item()} | norm: {comp_grad.norm().item():.4f}")
    else:
        print("Linear(150,4) [state_compression] grad exists: NO")

    # Check Quantum Circuit Parameters
    # In the mock, Op1QAllLayer might not have params properly linked to loss unless we used them.
    # In real TorchQuantum, they are nn.Parameters.
    q_param_grad = actor.rx_layers[0].params.grad if actor.rx_layers[0].has_params and hasattr(actor.rx_layers[0], 'params') else None
    if q_param_grad is not None:
        print(f"Quantum Parameter (Layer 0 RX) grad exists: YES | finite: {torch.isfinite(q_param_grad).all().item()} | norm: {q_param_grad.norm().item():.4f}")
    else:
        print("Quantum Parameter (Layer 0 RX) grad exists: NO")
        
    # Check Linear(4,5)
    exp_grad = actor.action_expansion[0].weight.grad
    if exp_grad is not None:
        print(f"Linear(4,5) [action_expansion] grad exists: YES | finite: {torch.isfinite(exp_grad).all().item()} | norm: {exp_grad.norm().item():.4f}")
    else:
        print("Linear(4,5) [action_expansion] grad exists: NO")

def test_ddpg_update():
    print("\n==================================================")
    print("DDPG UPDATE TEST")
    print("==================================================")
    
    agent = DDPGAgent(config={'gamma': 0.95})
    buf = ReplayBuffer(150, 5, max_size=100)
    
    # Populate replay with 40 transitions
    for _ in range(40):
        buf.add(np.random.randn(150), np.random.randn(5), 1.0, np.random.randn(150), False)
        
    # Check initial parameters
    actor_initial = agent.actor.state_compression[0].weight.clone()
    critic_initial = agent.critic.net[0].weight.clone()
    
    actor_loss, critic_loss = agent.train(buf, batch_size=32)
    
    actor_new = agent.actor.state_compression[0].weight
    critic_new = agent.critic.net[0].weight
    
    actor_changed = not torch.equal(actor_initial, actor_new)
    critic_changed = not torch.equal(critic_initial, critic_new)
    
    print(f"Actor Loss: {actor_loss:.4f} (finite: {np.isfinite(actor_loss)})")
    print(f"Critic Loss: {critic_loss:.4f} (finite: {np.isfinite(critic_loss)})")
    print(f"Actor Parameters Changed: {actor_changed}")
    print(f"Critic Parameters Changed: {critic_changed}")

if __name__ == "__main__":
    test_qnn_gradients()
    test_ddpg_update()
