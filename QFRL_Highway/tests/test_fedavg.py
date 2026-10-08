import pytest
import torch
import numpy as np
from federated.client import FederatedClient
from federated.server import FederatedServer
from federated.fedavg import fedavg

def test_fedavg_mathematical_validation():
    # Deterministic toy test
    c1 = {
        'params': {
            'actor': {'layer1': torch.tensor([1.0, 1.0])},
            'critic': {'layer1': torch.tensor([1.0, 1.0])}
        },
        'num_samples': 10
    }
    c2 = {
        'params': {
            'actor': {'layer1': torch.tensor([3.0, 3.0])},
            'critic': {'layer1': torch.tensor([3.0, 3.0])}
        },
        'num_samples': 20
    }
    c3 = {
        'params': {
            'actor': {'layer1': torch.tensor([5.0, 5.0])},
            'critic': {'layer1': torch.tensor([5.0, 5.0])}
        },
        'num_samples': 30
    }
    
    global_params = fedavg([c1, c2, c3])
    expected = (10*1.0 + 20*3.0 + 30*5.0) / 60.0
    
    assert torch.allclose(global_params['actor']['layer1'], torch.tensor([expected, expected]))
    assert torch.allclose(global_params['critic']['layer1'], torch.tensor([expected, expected]))

def test_full_federated_flow():
    config = {
        'vehicles': {'count': 5},
        'tasks': {'input_bits': 1000, 'cycles_per_bit': 10},
        'replay_size': 100
    }
    
    server = FederatedServer(num_clients=3)
    clients = [FederatedClient(client_id=i, base_seed=42, config=config) for i in range(3)]
    
    # Verify identical initial global models
    c0_params = clients[0].get_parameters()
    server.initialize_global_model(c0_params)
    
    for c in clients:
        c.set_parameters(server.get_global_parameters())
        
        # Verify they are identical
        c_params = c.get_parameters()
        for k in c0_params['actor']:
            assert torch.allclose(c0_params['actor'][k], c_params['actor'][k])

    # Change one client's parameters artificially
    clients[0].actor.state_compression[0].weight.data.fill_(1.0)
    
    # Verify local training changes local parameters
    clients[1].collect_experience(steps=5)
    clients[1].local_train(local_steps=2, batch_size=2)
    
    c1_params_after = clients[1].get_parameters()
    is_changed = False
    for k in c0_params['actor']:
        if not torch.allclose(c0_params['actor'][k], c1_params_after['actor'][k]):
            is_changed = True
            break
    assert is_changed, "Local training should change parameters."
    
    # Run two federated rounds
    for round_num in range(2):
        client_states = []
        for c in clients:
            c.collect_experience(steps=2)
            c.local_train(local_steps=1, batch_size=2)
            client_states.append(c.get_model_state())
            
        server.aggregate(client_states)
        global_params = server.get_global_parameters()
        
        # Verify client can receive global parameters
        for c in clients:
            c.set_parameters(global_params)
            
        # Verify all parameters remain finite
        for k, v in global_params['actor'].items():
            assert torch.isfinite(v).all()
        for k, v in global_params['critic'].items():
            assert torch.isfinite(v).all()

    # Verify actor and critic architectures remain unchanged
    c0_final = clients[0].get_parameters()
    assert list(c0_final['actor'].keys()) == list(c0_params['actor'].keys())
    assert list(c0_final['critic'].keys()) == list(c0_params['critic'].keys())
