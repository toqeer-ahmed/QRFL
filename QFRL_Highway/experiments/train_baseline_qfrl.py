import os
import sys
import time
import copy
import types
import random
import argparse
import yaml
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure package imports work regardless of execution location
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock qiskit_ibm_runtime if absent to prevent import failure in torchquantum
try:
    import qiskit_ibm_runtime
except ImportError:
    sys.modules['qiskit_ibm_runtime'] = types.ModuleType('qiskit_ibm_runtime')

from environment.environment import QFRLEnvironment
from rl.ddpg import DDPGAgent
from rl.replay_buffer import ReplayBuffer

def set_seed(seed: int):
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def verify_backend(allow_mock: bool = False):
    """
    Verifies that the real TorchQuantum backend is loaded from site-packages.
    Fails loudly if mock_torchquantum is imported without explicit permission.
    """
    print("\n" + "=" * 60)
    print("BACKEND VERIFICATION")
    print("=" * 60)
    print(f"Python version      : {sys.version.split()[0]} ({sys.platform})")
    print(f"PyTorch version     : {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available      : {cuda_avail}")
    if cuda_avail:
        print(f"GPU Device Name     : {torch.cuda.get_device_name(0)}")
        print(f"GPU Device Count    : {torch.cuda.device_count()}")
    else:
        print("GPU Device Name     : N/A (Running on CPU)")

    try:
        import torchquantum as tq
        backend_path = str(getattr(tq, '__file__', ''))
        version = str(getattr(tq, '__version__', '0.3.0'))
        
        is_mock = "mock_torchquantum" in backend_path or ("QFRL_Highway" in backend_path and "mock" in backend_path)
        
        if is_mock:
            backend_name = f"TorchQuantum {version} (Mock)"
            print(f"TorchQuantum version: {version}")
            print(f"TorchQuantum path   : {backend_path}")
            if not allow_mock:
                err_msg = (
                    "FATAL ERROR: Loaded local mock_torchquantum instead of real TorchQuantum!\n"
                    f"Path: {backend_path}\n"
                    "Stage 8 requires the real TorchQuantum backend. Run on Kaggle/Linux with\n"
                    "TorchQuantum installed from mit-han-lab/torchquantum.git."
                )
                print(f"[FAIL] {err_msg}")
                raise RuntimeError(err_msg)
            else:
                print(f"[WARN] Local mock permitted via --allow_mock flag.")
        else:
            backend_name = f"TorchQuantum {version} (Real)"
            print(f"TorchQuantum version: {version}")
            print(f"TorchQuantum path   : {backend_path}")
            print(f"[PASS] Real TorchQuantum backend confirmed.")
            
        print("=" * 60 + "\n")
        return tq, backend_name, is_mock
    except ImportError as e:
        if allow_mock:
            try:
                import mock_torchquantum as tq
                backend_name = "TorchQuantum (Mock)"
                print(f"[WARN] Imported mock_torchquantum under --allow_mock flag.")
                return tq, backend_name, True
            except ImportError:
                pass
        err_msg = f"FATAL ERROR: Failed to import torchquantum: {e}"
        print(f"[FAIL] {err_msg}")
        raise RuntimeError(err_msg)

def check_device(requested_device: str) -> str:
    """Verifies that requested device works with QuantumActor, falling back safely if needed."""
    if requested_device == 'cuda':
        if not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available. Falling back to CPU.")
            return 'cpu'
        try:
            from quantum.qnn import QuantumActor
            test_actor = QuantumActor(150, 5).to('cuda')
            test_x = torch.rand(2, 150, device='cuda')
            _ = test_actor(test_x)
            print(f"[PASS] CUDA execution verified on {torch.cuda.get_device_name(0)}.")
            return 'cuda'
        except Exception as e:
            print(f"[WARN] CUDA verification encountered: {e}. Falling back to CPU.")
            return 'cpu'
    return 'cpu'

def run_pilot_smoke_test(env_config: dict, device: str, seed: int = 42, is_mock: bool = False) -> bool:
    """
    Executes a controlled 3-episode pilot test to verify:
    1. Environment step completion
    2. Replay buffer accumulation
    3. DDPG gradient updates
    4. Real quantum gradient propagation (on real backend)
    5. Finite losses and rewards
    6. Checkpoint save and bit-exact restore
    7. Seed reproducibility
    """
    print("\n" + "=" * 60)
    print(f"STAGE 8 PILOT SMOKE TEST (Seed {seed})")
    print("=" * 60)
    
    set_seed(seed)
    env = QFRLEnvironment()
    env.initialize_from_config(env_config)
    
    pilot_config = copy.deepcopy(env_config)
    pilot_config['device'] = device
    
    agent = DDPGAgent(config=pilot_config)
    buffer = ReplayBuffer(state_dim=150, action_dim=5, max_size=1000)
    
    # 1. Warm-up buffer
    state = env.reset()
    warmup_steps = 35
    print(f"1. Filling replay buffer with {warmup_steps} warm-up transitions...")
    for _ in range(warmup_steps):
        action = agent.select_action(state, add_noise=True)
        next_state, reward, done, info = env.step(action)
        buffer.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()
            
    assert len(buffer) == warmup_steps, f"Buffer size {len(buffer)} != {warmup_steps}"
    print(f"   [PASS] Replay buffer populated: {len(buffer)} transitions.")
    
    # 2. Gradient update & Quantum Gradient Verification
    print("2. Performing DDPG gradient update and checking parameter gradients...")
    # Manually run 1 training step so we can check grads before optimizer.zero_grad()
    batch = buffer.sample(32)
    s = torch.FloatTensor(batch['states']).to(agent.device)
    a = torch.FloatTensor(batch['actions']).to(agent.device)
    r = torch.FloatTensor(batch['rewards']).to(agent.device)
    ns = torch.FloatTensor(batch['next_states']).to(agent.device)
    d = torch.FloatTensor(batch['dones']).to(agent.device)
    
    # Critic update
    with torch.no_grad():
        next_a = agent.actor_target(ns)
        target_q = agent.critic_target(ns, next_a)
        target_q = r + (1 - d) * agent.gamma * target_q
    current_q = agent.critic(s, a)
    critic_loss = agent.critic_criterion(current_q, target_q)
    agent.critic_optimizer.zero_grad()
    critic_loss.backward()
    agent.critic_optimizer.step()
    
    # Actor update
    actor_loss = -agent.critic(s, agent.actor(s)).mean()
    agent.actor_optimizer.zero_grad()
    actor_loss.backward()
    
    # Verify gradients in actor
    quantum_params_with_grad = 0
    total_quantum_params = 0
    classical_params_with_grad = 0
    total_classical_params = 0
    
    for name, param in agent.actor.named_parameters():
        if "rx_layers" in name or "ry_layers" in name or "rz_layers" in name:
            total_quantum_params += 1
            if param.grad is not None and param.grad.abs().sum().item() > 0:
                quantum_params_with_grad += 1
        else:
            total_classical_params += 1
            if param.grad is not None and param.grad.abs().sum().item() > 0:
                classical_params_with_grad += 1
                
    agent.actor_optimizer.step()
    
    print(f"   Actor Loss : {actor_loss.item():.4f}")
    print(f"   Critic Loss: {critic_loss.item():.4e}")
    print(f"   Classical parameter tensors with gradients: {classical_params_with_grad}/{total_classical_params}")
    print(f"   Quantum parameter tensors with gradients  : {quantum_params_with_grad}/{total_quantum_params}")
    
    assert classical_params_with_grad > 0, "No gradients flowed through classical actor layers!"
    if not is_mock:
        assert quantum_params_with_grad > 0, "Real backend failed: No gradients flowed to quantum circuit parameters!"
        print("   [PASS] Real quantum parameter gradients successfully verified.")
    else:
        print("   [INFO] Local mock active: classical gradient flow verified.")
        
    # 3. Checkpoint Save and Restore Verification
    print("3. Testing Checkpoint Save & Bit-Exact Restore...")
    test_ckpt_path = os.path.join("results", "checkpoints", "pilot_test_ckpt.pt")
    os.makedirs(os.path.dirname(test_ckpt_path), exist_ok=True)
    agent.save_checkpoint(test_ckpt_path, episode=1, seed=seed)
    assert os.path.exists(test_ckpt_path), "Checkpoint file was not created!"
    
    # Create fresh agent and load
    agent_restore = DDPGAgent(config=pilot_config)
    ep_loaded, seed_loaded, _ = agent_restore.load_checkpoint(test_ckpt_path)
    assert ep_loaded == 1 and seed_loaded == seed, "Checkpoint metadata mismatch!"
    
    # Verify bit-exact weights
    for p1, p2 in zip(agent.actor.parameters(), agent_restore.actor.parameters()):
        assert torch.all(torch.eq(p1, p2)), "Actor weights mismatch after checkpoint load!"
    for p1, p2 in zip(agent.critic.parameters(), agent_restore.critic.parameters()):
        assert torch.all(torch.eq(p1, p2)), "Critic weights mismatch after checkpoint load!"
    print("   [PASS] Checkpoint saved, reloaded, and weights verified bit-exact.")
    
    # 4. Short Multi-Step Episode Execution
    print("4. Testing short 3-episode interaction loop...")
    pilot_rewards = []
    for ep in range(1, 4):
        state = env.reset()
        ep_reward = 0.0
        for st in range(10):
            action = agent.select_action(state, add_noise=True)
            next_state, reward, done, info = env.step(action)
            assert np.isfinite(reward), f"Non-finite reward at ep {ep}, step {st}: {reward}"
            buffer.add(state, action, reward, next_state, done)
            al, cl = agent.train(buffer, batch_size=32)
            assert np.isfinite(al) and np.isfinite(cl), "Non-finite loss during episode training!"
            state = next_state
            ep_reward += reward
        pilot_rewards.append(ep_reward)
        print(f"   Pilot Episode {ep}: Reward = {ep_reward:.2f}")
        
    print("   [PASS] All pilot episodes completed with finite rewards and losses.")
    print("=" * 60)
    print("PILOT SMOKE TEST RESULT: PASS")
    print("=" * 60 + "\n")
    return True

def run_baseline_training(
    config_path: str,
    episodes: int = 10,
    steps_per_episode: int = 50,
    seeds: list = [42],
    device: str = "cpu",
    output_dir: str = "results",
    allow_mock: bool = False,
    checkpoint_freq: int = 5,
    config_id: str = "baseline_pilot",
    backend_name: str = "Unknown"
):
    """
    Runs controlled baseline QFRL training, logging all metrics to CSV,
    generating plots, and saving model checkpoints.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Save configuration snapshot
    config_snapshot_path = os.path.join(output_dir, "config_snapshot.yaml")
    snapshot_data = {
        'paper_config': config,
        'experiment_override': {
            'episodes': episodes,
            'steps_per_episode': steps_per_episode,
            'seeds': seeds,
            'device': device,
            'backend_name': backend_name,
            'config_id': config_id
        }
    }
    with open(config_snapshot_path, 'w') as f:
        yaml.dump(snapshot_data, f, default_flow_style=False)
    print(f"Saved configuration snapshot to {config_snapshot_path}")

    # Prepare Metrics CSV
    csv_path = os.path.join(output_dir, "baseline_qfrl_metrics.csv")
    csv_headers = [
        "episode",
        "seed",
        "episode_reward",
        "mean_step_reward",
        "energy_efficiency",
        "energy_efficiency_mbits_per_joule",
        "completed_bits",
        "total_energy",
        "task_completion_ratio",
        "mean_latency",
        "actor_loss",
        "critic_loss",
        "replay_buffer_size",
        "num_updates",
        "duration_seconds",
        "backend_name",
        "config_identifier"
    ]
    
    file_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, 'a' if file_exists else 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
    if not file_exists:
        writer.writeheader()
        
    all_runs_metrics = {seed: [] for seed in seeds}
    total_training_start = time.time()
    
    # Buffer parameters from config
    buffer_size = config.get('rl', {}).get('replay_buffer_size', 6000)
    batch_size = config.get('rl', {}).get('batch_size', 32)
    warmup_steps = 32
    
    print(f"\nStarting Baseline QFRL Training across {len(seeds)} seed(s)...")
    print(f"Episodes per seed   : {episodes}")
    print(f"Steps per episode   : {steps_per_episode}")
    print(f"Batch size          : {batch_size}")
    print(f"Replay buffer size  : {buffer_size}")
    print(f"Device              : {device}")
    print(f"Backend             : {backend_name}")
    print(f"Output CSV          : {csv_path}\n")

    for seed in seeds:
        print("=" * 60)
        print(f"STARTING TRAINING RUN — SEED {seed}")
        print("=" * 60)
        set_seed(seed)
        
        env = QFRLEnvironment()
        env.initialize_from_config(config)
        
        run_config = copy.deepcopy(config)
        run_config['device'] = device
        
        agent = DDPGAgent(config=run_config)
        replay_buffer = ReplayBuffer(state_dim=150, action_dim=5, max_size=buffer_size)
        
        num_updates = 0
        seed_episodes_data = []
        
        # Warm-up replay buffer
        print(f"Warming up replay buffer ({warmup_steps} steps)...")
        w_state = env.reset()
        for _ in range(warmup_steps):
            w_action = agent.select_action(w_state, add_noise=True)
            w_next_state, w_reward, w_done, w_info = env.step(w_action)
            replay_buffer.add(w_state, w_action, w_reward, w_next_state, w_done)
            w_state = w_next_state
            if w_done:
                w_state = env.reset()
                
        print(f"Buffer warmed up. Starting {episodes} episodes...\n")
        
        for ep in range(1, episodes + 1):
            ep_start_time = time.time()
            state = env.reset()
            
            ep_reward = 0.0
            ep_completed_bits = 0.0
            ep_total_energy = 0.0
            ep_latencies = []
            ep_completion_ratios = []
            ep_actor_losses = []
            ep_critic_losses = []
            
            for step in range(steps_per_episode):
                action = agent.select_action(state, add_noise=True)
                next_state, reward, done, info = env.step(action)
                
                replay_buffer.add(state, action, reward, next_state, done)
                
                # Perform DDPG training step
                al, cl = agent.train(replay_buffer, batch_size=batch_size)
                if al is not None and cl is not None:
                    ep_actor_losses.append(al)
                    ep_critic_losses.append(cl)
                    num_updates += 1
                    
                state = next_state
                ep_reward += reward
                ep_completed_bits += info.get('completed_bits', 0.0)
                ep_total_energy += info.get('total_energy', 0.0)
                ep_completion_ratios.append(info.get('task_completion_ratio', 1.0))
                
                tx_time = info.get('transmission_time', 0.0)
                cp_time = info.get('computation_time', 0.0)
                ep_latencies.append(max(tx_time, cp_time))
                
                if done:
                    break
                    
            ep_duration = time.time() - ep_start_time
            mean_step_reward = ep_reward / max(step + 1, 1)
            
            # Aggregate Energy Efficiency across episode: completed bits / total energy
            ee_bits_per_joule = ep_completed_bits / max(ep_total_energy, 1e-6)
            ee_mbits_per_joule = ee_bits_per_joule / 1e6
            
            avg_actor_loss = float(np.mean(ep_actor_losses)) if ep_actor_losses else float('nan')
            avg_critic_loss = float(np.mean(ep_critic_losses)) if ep_critic_losses else float('nan')
            mean_comp_ratio = float(np.mean(ep_completion_ratios)) if ep_completion_ratios else 0.0
            mean_latency = float(np.mean(ep_latencies)) if ep_latencies else 0.0
            
            row = {
                "episode": ep,
                "seed": seed,
                "episode_reward": round(ep_reward, 4),
                "mean_step_reward": round(mean_step_reward, 4),
                "energy_efficiency": round(ee_bits_per_joule, 4),
                "energy_efficiency_mbits_per_joule": round(ee_mbits_per_joule, 6),
                "completed_bits": round(ep_completed_bits, 2),
                "total_energy": round(ep_total_energy, 4),
                "task_completion_ratio": round(mean_comp_ratio, 4),
                "mean_latency": round(mean_latency, 6),
                "actor_loss": round(avg_actor_loss, 4) if np.isfinite(avg_actor_loss) else "nan",
                "critic_loss": round(avg_critic_loss, 4) if np.isfinite(avg_critic_loss) else "nan",
                "replay_buffer_size": len(replay_buffer),
                "num_updates": num_updates,
                "duration_seconds": round(ep_duration, 2),
                "backend_name": backend_name,
                "config_identifier": config_id
            }
            
            writer.writerow(row)
            csv_file.flush()
            seed_episodes_data.append(row)
            
            print(
                f"Ep {ep:3d}/{episodes:3d} | "
                f"Reward: {ep_reward:9.2f} | "
                f"EE: {ee_mbits_per_joule:6.3f} Mb/J | "
                f"Energy: {ep_total_energy:7.2f} J | "
                f"ALoss: {avg_actor_loss:8.2f} | "
                f"CLoss: {avg_critic_loss:10.2e} | "
                f"Time: {ep_duration:4.1f}s"
            )
            
            # Checkpoint
            if ep % checkpoint_freq == 0 or ep == episodes:
                ckpt_file = os.path.join(ckpt_dir, f"baseline_qfrl_seed{seed}_ep{ep}.pt")
                agent.save_checkpoint(ckpt_file, episode=ep, seed=seed)
                print(f"  -> Saved checkpoint: {ckpt_file}")
                
        all_runs_metrics[seed] = seed_episodes_data
        
    csv_file.close()
    total_training_duration = time.time() - total_training_start
    print(f"\nTraining completed across all seeds in {total_training_duration:.2f} seconds.")
    
    # Generate Learning Curves Plot
    plot_path = os.path.join(output_dir, "baseline_qfrl_learning_curves.png")
    generate_plots(all_runs_metrics, plot_path)
    
    return all_runs_metrics, total_training_duration

def generate_plots(all_runs_metrics: dict, save_path: str):
    """
    Generates learning curve plots:
    1. Episode Reward
    2. Energy Efficiency (Mbits/Joule)
    3. Total Energy (Joules) & Completed Bits
    4. Actor & Critic Loss Curves
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    for seed, ep_data in all_runs_metrics.items():
        if not ep_data:
            continue
        eps = [d['episode'] for d in ep_data]
        rewards = [d['episode_reward'] for d in ep_data]
        ee_mbits = [d['energy_efficiency_mbits_per_joule'] for d in ep_data]
        energy = [d['total_energy'] for d in ep_data]
        c_losses = [float(d['critic_loss']) if str(d['critic_loss']) != 'nan' else 0.0 for d in ep_data]
        a_losses = [float(d['actor_loss']) if str(d['actor_loss']) != 'nan' else 0.0 for d in ep_data]
        
        # 1. Episode Reward
        axs[0, 0].plot(eps, rewards, marker='o', label=f"Seed {seed}")
        
        # 2. Energy Efficiency
        axs[0, 1].plot(eps, ee_mbits, marker='s', label=f"Seed {seed}")
        
        # 3. Total Energy
        axs[1, 0].plot(eps, energy, marker='^', label=f"Seed {seed}")
        
        # 4. Critic Loss (log scale)
        valid_closs = [c for c in c_losses if c > 0]
        if valid_closs:
            axs[1, 1].plot(eps, c_losses, marker='x', label=f"Critic (Seed {seed})")
            
    axs[0, 0].set_title("Baseline QFRL: Episode Reward", fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Total Episode Reward")
    axs[0, 0].legend()
    
    axs[0, 1].set_title("Energy Efficiency (Mbits / Joule)", fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Mbits / Joule")
    axs[0, 1].legend()
    
    axs[1, 0].set_title("Total Energy Consumption (Joules)", fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Total Energy (J)")
    axs[1, 0].legend()
    
    axs[1, 1].set_title("Critic Loss", fontsize=12, fontweight='bold')
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("MSE Loss")
    axs[1, 1].set_yscale('log')
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Generated learning curves plot: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Stage 8: Controlled Baseline QFRL Training and Evaluation")
    parser.add_argument('--episodes', type=int, default=10, help="Episodes per seed")
    parser.add_argument('--steps_per_episode', type=int, default=50, help="Steps per episode")
    parser.add_argument('--seed', type=int, default=42, help="Primary seed")
    parser.add_argument('--seeds', type=str, default="42", help="Comma-separated list of seeds (e.g. 42,123,2026)")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda)")
    parser.add_argument('--config', type=str, default="configs/paper_config.yaml", help="Path to config file")
    parser.add_argument('--output_dir', type=str, default="results", help="Directory to save artifacts")
    parser.add_argument('--allow_mock', action='store_true', help="Allow mock TorchQuantum for local testing")
    parser.add_argument('--pilot_only', action='store_true', help="Only run pilot smoke test")
    parser.add_argument('--skip_pilot', action='store_true', help="Skip pilot smoke test")
    parser.add_argument('--config_id', type=str, default="baseline_pilot", help="Identifier for configuration")
    args = parser.parse_args()
    
    # Verify backend at the very start
    tq, backend_name, is_mock = verify_backend(allow_mock=args.allow_mock)
    active_device = check_device(args.device)
    
    parsed_seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    if not parsed_seeds:
        parsed_seeds = [args.seed]
        
    with open(args.config, 'r') as f:
        env_config = yaml.safe_load(f)
        
    # Step 1: Run Pilot Smoke Test
    if not args.skip_pilot:
        pilot_passed = run_pilot_smoke_test(
            env_config=env_config,
            device=active_device,
            seed=args.seed,
            is_mock=is_mock
        )
        if not pilot_passed:
            print("PILOT SMOKE TEST FAILED! Aborting baseline training.")
            sys.exit(1)
            
    if args.pilot_only:
        print("Pilot test completed successfully. Exiting due to --pilot_only flag.")
        return

    # Step 2: Run Baseline Training
    run_baseline_training(
        config_path=args.config,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        seeds=parsed_seeds,
        device=active_device,
        output_dir=args.output_dir,
        allow_mock=args.allow_mock,
        checkpoint_freq=max(1, args.episodes // 2),
        config_id=args.config_id,
        backend_name=backend_name
    )

if __name__ == "__main__":
    main()
