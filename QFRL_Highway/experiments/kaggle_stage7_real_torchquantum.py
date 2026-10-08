import sys
import torch
import warnings

def run_environment_check():
    print("==================================================")
    print("1. ENVIRONMENT CHECK")
    print("==================================================")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA availability: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
    try:
        import torchquantum as tq
        print(f"TorchQuantum version: {getattr(tq, '__version__', 'Unknown')}")
        print(f"TorchQuantum import path: {tq.__file__}")
        
        # Explicit validation
        if "QFRL_Highway" in str(tq.__file__) and "mock_torchquantum" in str(tq.__file__):
            print("ERROR: Loaded local mock_torchquantum instead of real torchquantum!")
            sys.exit(1)
        if "QFRL_Highway" in str(tq.__file__) and "torchquantum" in str(tq.__file__):
            print("ERROR: Loaded local torchquantum instead of real torchquantum!")
            sys.exit(1)
            
    except ImportError as e:
        print("TorchQuantum is not installed or import failed.")
        print(e)
        sys.exit(1)

def run_qnn_forward_test():
    print("\n==================================================")
    print("2. REAL QNN FORWARD TEST")
    print("==================================================")
    from quantum.qnn import QuantumActor
    actor = QuantumActor(state_dim=150, action_dim=5)
    
    for batch_size in [1, 4, 32]:
        x = torch.rand(batch_size, 150)
        out = actor(x)
        print(f"Batch size: {batch_size}")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {out.shape}")
        print(f"  Is finite: {torch.isfinite(out).all().item()}")
        
def run_gradient_test():
    print("\n==================================================")
    print("3. REAL GRADIENT TEST")
    print("==================================================")
    from quantum.qnn import QuantumActor
    import torch.optim as optim
    
    actor = QuantumActor(state_dim=150, action_dim=5)
    opt = optim.Adam(actor.parameters(), lr=0.01)
    
    x = torch.rand(32, 150)
    target = torch.rand(32, 5)
    
    out = actor(x)
    loss = torch.nn.functional.mse_loss(out, target)
    
    opt.zero_grad()
    loss.backward()
    
    # Check gradients
    print("Gradient Norms:")
    for name, param in actor.named_parameters():
        if param.grad is not None:
            print(f"  {name}: {param.grad.norm().item():.6f}")
        else:
            print(f"  {name}: None")

def run_shot_test():
    print("\n==================================================")
    print("4. SHOT-BASED MEASUREMENT TEST")
    print("==================================================")
    try:
        import torchquantum as tq
        # Try to execute a simple circuit with shots
        qdev = tq.QuantumDevice(n_wires=2)
        qdev.h(wires=0)
        qdev.cnot(wires=[0, 1])
        # TorchQuantum typically uses expectation by default.
        # To use shots, one might call measure with shots=2048.
        # We will attempt to call measure.
        res = tq.measure(qdev, n_shots=2048)
        print("Shot measurement succeeded!")
        print("Result keys (e.g. state counts):", res)
    except Exception as e:
        print("Shot measurement test failed or unsupported in this way.")
        print(e)

def run_ddpg_update():
    print("\n==================================================")
    print("5. REAL DDPG UPDATE")
    print("==================================================")
    from rl.ddpg import DDPGAgent
    from rl.replay_buffer import ReplayBuffer
    
    agent = DDPGAgent(state_dim=150, action_dim=5)
    buffer = ReplayBuffer(150, 5, max_size=100)
    
    for _ in range(32):
        s = np.random.rand(150)
        a = np.random.rand(5)
        r = np.random.rand()
        ns = np.random.rand(150)
        d = False
        buffer.add(s, a, r, ns, d)
        
    actor_loss, critic_loss = agent.train(buffer, batch_size=32)
    print(f"Actor loss: {actor_loss}")
    print(f"Critic loss: {critic_loss}")

def run_fedavg_compatibility():
    print("\n==================================================")
    print("6. FEDAVG COMPATIBILITY")
    print("==================================================")
    from federated.server import FederatedServer
    from federated.client import FederatedClient
    import yaml
    
    with open('configs/paper_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    server = FederatedServer(2)
    c1 = FederatedClient(0, 42, config)
    c2 = FederatedClient(1, 42, config)
    
    server.initialize_global_model(c1.get_parameters())
    c1.set_parameters(server.get_global_parameters())
    c2.set_parameters(server.get_global_parameters())
    
    # Run local train
    c1.collect_experience(steps=32)
    c1.local_train(local_steps=1, batch_size=32)
    
    c2.collect_experience(steps=32)
    c2.local_train(local_steps=1, batch_size=32)
    
    states = [c1.get_model_state(), c2.get_model_state()]
    server.aggregate(states)
    
    c1.set_parameters(server.get_global_parameters())
    print("FedAvg Compatibility Verified.")

if __name__ == "__main__":
    import sys
    sys.path.append(".")  # ensure local modules can be imported
    run_environment_check()
    run_qnn_forward_test()
    run_gradient_test()
    run_shot_test()
    run_ddpg_update()
    run_fedavg_compatibility()
    print("\nAll tests completed.")
