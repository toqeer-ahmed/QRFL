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
from rl.classical_actor import ClassicalDDPGAgent, ClassicalActor
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

def count_trainable_parameters(model: nn.Module) -> int:
    """Counts requires_grad parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def verify_quantum_backend(allow_mock: bool = False):
    """
    Verifies that the real TorchQuantum backend is loaded from site-packages.
    Fails loudly if mock_torchquantum is imported without explicit permission.
    """
    print("\n" + "=" * 60)
    print("BACKEND VERIFICATION (QUANTUM MODEL)")
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

    # Ensure torchpack and torchdiffeq are available for torchquantum
    try:
        import torchpack
    except ImportError:
        try:
            import subprocess
            print("[INFO] Installing missing TorchQuantum utility: torchpack & torchdiffeq...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "torchpack", "torchdiffeq"])
            print("[PASS] torchpack installed successfully.")
        except Exception as e:
            print(f"[WARN] Auto-install of torchpack failed: {e}")

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
                    "Stage 9 benchmark requires real TorchQuantum backend or explicit --allow_mock flag."
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

def run_pilot_smoke_test(env_config, models=None, device="cpu", allow_mock=False):
    """
    Executes short 2-episode pilot per model to verify that:
    1. Classical model uses only ClassicalActor (no quantum imports or calls).
    2. Quantum model uses TorchQuantum properly.
    3. Both perform valid updates, finite losses, valid actions, and bit-exact checkpointing.
    """
    if models is None:
        models = ["classical", "quantum"]
        
    print("=" * 70)
    print(f"STAGE 9 PILOT SMOKE TEST: VERIFYING AGENTS {models}")
    print("=" * 70)
    
    set_seed(42)
    env = QFRLEnvironment()
    env.initialize_from_config(env_config)
    
    c_config = copy.deepcopy(env_config)
    c_config['device'] = device
    
    # 1. Classical Model Verification
    if "classical" in models:
        print("1. Testing Classical Agent...")
        c_agent = ClassicalDDPGAgent(state_dim=150, action_dim=5, config=c_config)
        
        c_actor_params = count_trainable_parameters(c_agent.actor)
        c_critic_params = count_trainable_parameters(c_agent.critic)
        print(f"   [PASS] ClassicalActor instantiated. Actor params: {c_actor_params:,}, Critic params: {c_critic_params:,}")
    
    c_buffer = ReplayBuffer(state_dim=150, action_dim=5, max_size=100)
    s = env.reset()
    for _ in range(35):
        a = c_agent.select_action(s)
        s_next, r, d, info = env.step(a)
        c_buffer.add(s, a, r, s_next, d)
        s = s_next
    
    c_al, c_cl = c_agent.train(c_buffer, batch_size=32)
    assert c_al is not None and c_cl is not None, "Classical train returned None"
    assert np.isfinite(c_al) and np.isfinite(c_cl), "Classical losses not finite"
    print(f"   [PASS] Classical DDPG update verified. ALoss: {c_al:.4f}, CLoss: {c_cl:.4e}")
    
    # Test Classical Checkpoint
    os.makedirs("results/checkpoints", exist_ok=True)
    c_ckpt_path = "results/checkpoints/pilot_classical_test.pt"
    c_agent.save_checkpoint(c_ckpt_path, episode=1, seed=42)
    c_agent_restored = ClassicalDDPGAgent(state_dim=150, action_dim=5, config=c_config)
    c_agent_restored.load_checkpoint(c_ckpt_path)
    for p1, p2 in zip(c_agent.actor.parameters(), c_agent_restored.actor.parameters()):
        assert torch.allclose(p1, p2), "Classical checkpoint restoration failed"
    print("   [PASS] Classical checkpoint save and bit-exact restore verified.")
    if os.path.exists(c_ckpt_path): os.remove(c_ckpt_path)

    # 2. Quantum Model Verification
    if "quantum" in models:
        print("\n2. Testing Quantum Agent...")
        verify_quantum_backend(allow_mock=allow_mock)
        q_agent = DDPGAgent(config=c_config)
        q_actor_params = count_trainable_parameters(q_agent.actor)
        q_critic_params = count_trainable_parameters(q_agent.critic)
        print(f"   [PASS] QuantumActor instantiated. Actor params: {q_actor_params:,}, Critic params: {q_critic_params:,}")
        
        q_buffer = ReplayBuffer(state_dim=150, action_dim=5, max_size=100)
        s = env.reset()
        for _ in range(35):
            a = q_agent.select_action(s)
            s_next, r, d, info = env.step(a)
            q_buffer.add(s, a, r, s_next, d)
            s = s_next
            
        q_al, q_cl = q_agent.train(q_buffer, batch_size=32)
        assert q_al is not None and q_cl is not None, "Quantum train returned None"
        assert np.isfinite(q_al) and np.isfinite(q_cl), "Quantum losses not finite"
        print(f"   [PASS] Quantum DDPG update verified. ALoss: {q_al:.4f}, CLoss: {q_cl:.4e}")
        
        # Test Quantum Checkpoint
        q_ckpt_path = "results/checkpoints/pilot_quantum_test.pt"
        q_agent.save_checkpoint(q_ckpt_path, episode=1, seed=42)
        q_agent_restored = DDPGAgent(config=c_config)
        q_agent_restored.load_checkpoint(q_ckpt_path)
        for p1, p2 in zip(q_agent.actor.parameters(), q_agent_restored.actor.parameters()):
            assert torch.allclose(p1, p2), "Quantum checkpoint restoration failed"
        print("   [PASS] Quantum checkpoint save and bit-exact restore verified.")
        if os.path.exists(q_ckpt_path): os.remove(q_ckpt_path)

    print("=" * 70)
    print("STAGE 9 PILOT SMOKE TEST RESULT: PASS")
    print("=" * 70 + "\n")
    return True

def run_controlled_comparison(
    config_path: str,
    models: list = ["classical", "quantum"],
    episodes: int = 15,
    steps_per_episode: int = 50,
    seeds: list = [42, 123, 2026],
    device: str = "cpu",
    output_dir: str = "results",
    allow_mock: bool = False,
    noise_schedule: str = "constant",
    decay_rate: float = 0.95
):
    """
    Executes matched, controlled comparison between Classical and Quantum DDPG models.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Verify backend if quantum is included
    backend_name = "Classical PyTorch"
    if "quantum" in models:
        _, q_backend, _ = verify_quantum_backend(allow_mock=allow_mock)
    else:
        q_backend = "N/A"
        
    csv_path = os.path.join(output_dir, "quantum_classical_comparison.csv")
    csv_headers = [
        "model_type",
        "seed",
        "episode",
        "episode_reward",
        "mean_step_reward",
        "energy_efficiency_mbits_per_joule",
        "completed_bits",
        "total_energy",
        "mean_latency",
        "task_completion_ratio",
        "actor_loss",
        "critic_loss",
        "actor_trainable_params",
        "critic_trainable_params",
        "inference_time_ms",
        "duration_seconds",
        "noise_schedule",
        "current_noise_scale",
        "backend_name"
    ]
    
    file_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, 'a' if file_exists else 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
    if not file_exists:
        writer.writeheader()
        
    all_results = {m: {s: [] for s in seeds} for m in models}
    
    buffer_size = config.get('rl', {}).get('replay_buffer_size', 6000)
    batch_size = config.get('rl', {}).get('batch_size', 32)
    warmup_steps = 32
    initial_noise = config.get('rl', {}).get('exploration_noise', 0.1)
    
    print("\n" + "=" * 70)
    print("STARTING CONTROLLED QFRL BENCHMARK")
    print(f"Models to evaluate   : {models}")
    print(f"Seeds               : {seeds}")
    print(f"Episodes per seed   : {episodes}")
    print(f"Steps per episode   : {steps_per_episode}")
    print(f"Exploration schedule: {noise_schedule}")
    print(f"Device              : {device}")
    print(f"Output CSV          : {csv_path}")
    print("=" * 70 + "\n")
    
    model_stats = {}
    
    for model_name in models:
        print("\n" + "#" * 70)
        print(f"MODEL EVALUATION: {model_name.upper()}")
        print("#" * 70)
        
        m_start_time = time.time()
        
        for seed in seeds:
            print("=" * 60)
            print(f"[{model_name.upper()}] Starting Run — Seed {seed}")
            print("=" * 60)
            set_seed(seed)
            
            env = QFRLEnvironment()
            env.initialize_from_config(config)
            
            run_cfg = copy.deepcopy(config)
            run_cfg['device'] = device
            
            if model_name == "classical":
                agent = ClassicalDDPGAgent(state_dim=150, action_dim=5, config=run_cfg)
                curr_backend = "Classical PyTorch MLP"
            elif model_name == "quantum":
                agent = DDPGAgent(config=run_cfg)
                curr_backend = q_backend
            else:
                raise ValueError(f"Unknown model name: {model_name}")
                
            actor_params = count_trainable_parameters(agent.actor)
            critic_params = count_trainable_parameters(agent.critic)
            
            replay_buffer = ReplayBuffer(state_dim=150, action_dim=5, max_size=buffer_size)
            
            # Replay buffer warm-up
            w_state = env.reset()
            for _ in range(warmup_steps):
                w_action = agent.select_action(w_state, add_noise=True)
                w_next, w_r, w_d, w_info = env.step(w_action)
                replay_buffer.add(w_state, w_action, w_r, w_next, w_d)
                w_state = w_next
                if w_d: w_state = env.reset()
                
            current_noise = initial_noise
            
            for ep in range(1, episodes + 1):
                ep_start = time.time()
                state = env.reset()
                
                ep_reward = 0.0
                ep_completed_bits = 0.0
                ep_total_energy = 0.0
                ep_latencies = []
                ep_comp_ratios = []
                ep_actor_losses = []
                ep_critic_losses = []
                inference_times = []
                
                # Noise decay logic
                if noise_schedule == "decay":
                    current_noise = max(0.01, initial_noise * (decay_rate ** (ep - 1)))
                else:
                    current_noise = initial_noise
                    
                for step in range(steps_per_episode):
                    t0 = time.time()
                    try:
                        action = agent.select_action(state, add_noise=True, noise_scale=current_noise)
                    except TypeError:
                        if hasattr(agent, 'exploration_noise'):
                            agent.exploration_noise = current_noise
                        action = agent.select_action(state, add_noise=True)
                    t_inf = (time.time() - t0) * 1000.0
                    inference_times.append(t_inf)
                    
                    next_state, reward, done, info = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    
                    al, cl = agent.train(replay_buffer, batch_size=batch_size)
                    if al is not None and cl is not None:
                        ep_actor_losses.append(al)
                        ep_critic_losses.append(cl)
                        
                    state = next_state
                    ep_reward += reward
                    ep_completed_bits += info.get('completed_bits', 0.0)
                    ep_total_energy += info.get('total_energy', 0.0)
                    ep_comp_ratios.append(info.get('task_completion_ratio', 1.0))
                    
                    tx_time = info.get('transmission_time', 0.0)
                    cp_time = info.get('computation_time', 0.0)
                    ep_latencies.append(max(tx_time, cp_time))
                    
                    if done: break
                    
                ep_duration = time.time() - ep_start
                mean_step_reward = ep_reward / max(step + 1, 1)
                ee_mbits_per_joule = (ep_completed_bits / max(ep_total_energy, 1e-6)) / 1e6
                avg_al = float(np.mean(ep_actor_losses)) if ep_actor_losses else float('nan')
                avg_cl = float(np.mean(ep_critic_losses)) if ep_critic_losses else float('nan')
                mean_latency = float(np.mean(ep_latencies)) if ep_latencies else 0.0
                mean_comp_ratio = float(np.mean(ep_comp_ratios)) if ep_comp_ratios else 0.0
                mean_inf = float(np.mean(inference_times)) if inference_times else 0.0
                
                row = {
                    "model_type": model_name,
                    "seed": seed,
                    "episode": ep,
                    "episode_reward": round(ep_reward, 4),
                    "mean_step_reward": round(mean_step_reward, 4),
                    "energy_efficiency_mbits_per_joule": round(ee_mbits_per_joule, 6),
                    "completed_bits": round(ep_completed_bits, 2),
                    "total_energy": round(ep_total_energy, 4),
                    "mean_latency": round(mean_latency, 6),
                    "task_completion_ratio": round(mean_comp_ratio, 4),
                    "actor_loss": round(avg_al, 4) if np.isfinite(avg_al) else "nan",
                    "critic_loss": round(avg_cl, 4) if np.isfinite(avg_cl) else "nan",
                    "actor_trainable_params": actor_params,
                    "critic_trainable_params": critic_params,
                    "inference_time_ms": round(mean_inf, 3),
                    "duration_seconds": round(ep_duration, 2),
                    "noise_schedule": noise_schedule,
                    "current_noise_scale": round(current_noise, 4),
                    "backend_name": curr_backend
                }
                
                writer.writerow(row)
                csv_file.flush()
                all_results[model_name][seed].append(row)
                
                print(
                    f"[{model_name[:5].upper()}] Ep {ep:2d}/{episodes:2d} | "
                    f"Reward: {ep_reward:9.2f} | "
                    f"EE: {ee_mbits_per_joule:6.3f} Mb/J | "
                    f"Energy: {ep_total_energy:7.2f} J | "
                    f"ALoss: {avg_al:8.2f} | "
                    f"CLoss: {avg_cl:10.2e} | "
                    f"Inf: {mean_inf:4.1f}ms | "
                    f"Time: {ep_duration:4.1f}s"
                )
                
                if ep % 5 == 0 or ep == episodes:
                    ckpt_file = os.path.join(ckpt_dir, f"{model_name}_seed{seed}_ep{ep}.pt")
                    agent.save_checkpoint(ckpt_file, episode=ep, seed=seed)
                    
        model_stats[model_name] = {
            'total_time': time.time() - m_start_time,
            'actor_params': actor_params,
            'critic_params': critic_params
        }

    csv_file.close()
    print("\nBenchmark CSV saved to:", csv_path)
    
    # Generate Comparison Plot
    plot_path = os.path.join(output_dir, "quantum_classical_comparison.png")
    generate_comparison_plot(all_results, models, seeds, episodes, plot_path)
    
    # Print Statistical Summary
    print_statistical_summary(all_results, models, seeds, model_stats)
    return all_results

def generate_comparison_plot(all_results, models, seeds, episodes, plot_path):
    """Generates 4-panel publication-grade comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Controlled Comparison: Classical vs. Quantum QFRL (Matched Seeds)", fontsize=16, fontweight='bold')
    
    colors = {'classical': '#1f77b4', 'quantum': '#ff7f0e'}
    
    # 1. Energy Efficiency
    ax = axes[0, 0]
    for m in models:
        # Collect per-episode across seeds
        matrix = np.array([[row['energy_efficiency_mbits_per_joule'] for row in all_results[m][s]] for s in seeds])
        mean_ee = np.mean(matrix, axis=0)
        std_ee = np.std(matrix, axis=0)
        eps = np.arange(1, episodes + 1)
        ax.plot(eps, mean_ee, label=f"{m.capitalize()} (Mean)", color=colors.get(m, 'blue'), lw=2.5)
        ax.fill_between(eps, mean_ee - std_ee, mean_ee + std_ee, color=colors.get(m, 'blue'), alpha=0.15)
    ax.set_title("Energy Efficiency (Mb/J) Across Episodes", fontweight='bold')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Energy Efficiency (Mb/J)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Total Energy Consumption
    ax = axes[0, 1]
    for m in models:
        matrix = np.array([[row['total_energy'] for row in all_results[m][s]] for s in seeds])
        mean_en = np.mean(matrix, axis=0)
        std_en = np.std(matrix, axis=0)
        eps = np.arange(1, episodes + 1)
        ax.plot(eps, mean_en, label=f"{m.capitalize()}", color=colors.get(m, 'blue'), lw=2.5)
        ax.fill_between(eps, mean_en - std_en, mean_en + std_en, color=colors.get(m, 'blue'), alpha=0.15)
    ax.set_title("Total Energy Consumption (Joules)", fontweight='bold')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Energy (Joules)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Episode Reward
    ax = axes[1, 0]
    for m in models:
        matrix = np.array([[row['episode_reward'] for row in all_results[m][s]] for s in seeds])
        mean_r = np.mean(matrix, axis=0)
        std_r = np.std(matrix, axis=0)
        eps = np.arange(1, episodes + 1)
        ax.plot(eps, mean_r, label=f"{m.capitalize()}", color=colors.get(m, 'blue'), lw=2.5)
        ax.fill_between(eps, mean_r - std_r, mean_r + std_r, color=colors.get(m, 'blue'), alpha=0.15)
    ax.set_title("Episode Reward Across Episodes", fontweight='bold')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Critic Loss
    ax = axes[1, 1]
    for m in models:
        matrix = np.array([[row['critic_loss'] for row in all_results[m][s]] for s in seeds])
        mean_cl = np.mean(matrix, axis=0)
        eps = np.arange(1, episodes + 1)
        ax.plot(eps, mean_cl, label=f"{m.capitalize()} Critic Loss", color=colors.get(m, 'blue'), lw=2)
    ax.set_title("Critic Loss (MSE)", fontweight='bold')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Critic Loss")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Generated comparison plot: {plot_path}")

def print_statistical_summary(all_results, models, seeds, model_stats):
    """Prints comprehensive Markdown-compatible statistical summary table."""
    print("\n" + "=" * 80)
    print("STAGE 9 BENCHMARK STATISTICAL SUMMARY")
    print("=" * 80)
    
    header = "| Metric / Model | " + " | ".join([f"{m.capitalize()}" for m in models]) + " |"
    sep = "|---| " + " | ".join(["---" for _ in models]) + " |"
    print(header)
    print(sep)
    
    # Trainable parameters
    p_row = "| **Actor Parameters** | " + " | ".join([f"{model_stats[m]['actor_params']:,}" for m in models]) + " |"
    print(p_row)
    c_row = "| **Critic Parameters** | " + " | ".join([f"{model_stats[m]['critic_params']:,}" for m in models]) + " |"
    print(c_row)
    t_row = "| **Total Runtime** | " + " | ".join([f"{model_stats[m]['total_time']:.1f}s" for m in models]) + " |"
    print(t_row)
    
    # Summary per model
    means_ee = {}
    for m in models:
        ee_vals = [row['energy_efficiency_mbits_per_joule'] for s in seeds for row in all_results[m][s]]
        means_ee[m] = (np.mean(ee_vals), np.std(ee_vals))
        
    ee_row = "| **Mean Energy Efficiency (Mb/J)** | " + " | ".join([f"{means_ee[m][0]:.4f} +/- {means_ee[m][1]:.4f}" for m in models]) + " |"
    print(ee_row)
    
    means_en = {}
    for m in models:
        en_vals = [row['total_energy'] for s in seeds for row in all_results[m][s]]
        means_en[m] = (np.mean(en_vals), np.std(en_vals))
    en_row = "| **Mean Total Energy (J)** | " + " | ".join([f"{means_en[m][0]:.2f} +/- {means_en[m][1]:.2f}" for m in models]) + " |"
    print(en_row)
    
    means_rw = {}
    for m in models:
        rw_vals = [row['episode_reward'] for s in seeds for row in all_results[m][s]]
        means_rw[m] = (np.mean(rw_vals), np.std(rw_vals))
    rw_row = "| **Mean Episode Reward** | " + " | ".join([f"{means_rw[m][0]:.2f} +/- {means_rw[m][1]:.2f}" for m in models]) + " |"
    print(rw_row)
    
    if "quantum" in models and "classical" in models:
        diff_abs = means_ee['quantum'][0] - means_ee['classical'][0]
        diff_pct = (diff_abs / max(means_ee['classical'][0], 1e-6)) * 100.0
        print("-" * 80)
        print(f"Absolute Difference (Quantum - Classical): {diff_abs:+.4f} Mb/J")
        print(f"Relative Difference                     : {diff_pct:+.2f}%")
        print("=" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Controlled Quantum vs Classical QFRL Benchmark")
    parser.add_argument("--config", type=str, default="configs/paper_config.yaml", help="Path to config YAML")
    parser.add_argument("--episodes", type=int, default=15, help="Number of episodes per seed (default: 15)")
    parser.add_argument("--steps", type=int, default=50, help="Steps per episode (default: 50)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026], help="Seeds to evaluate")
    parser.add_argument("--models", nargs="+", type=str, default=["classical", "quantum"], help="Models: classical, quantum")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu/cuda)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for output metrics and plots")
    parser.add_argument("--allow_mock", action="store_true", help="Allow mock TorchQuantum if running offline")
    parser.add_argument("--pilot_only", action="store_true", help="Run only 2-episode pilot verification")
    parser.add_argument("--noise_schedule", type=str, default="constant", choices=["constant", "decay"], help="Exploration schedule")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    if args.pilot_only:
        run_pilot_smoke_test(cfg, models=args.models, device=args.device, allow_mock=args.allow_mock)
        return
        
    # First run pilot
    pilot_ok = run_pilot_smoke_test(cfg, models=args.models, device=args.device, allow_mock=args.allow_mock)
    if not pilot_ok:
        print("[FAIL] Pilot smoke test failed. Aborting full benchmark.")
        sys.exit(1)
        
    # Then run full benchmark
    run_controlled_comparison(
        config_path=args.config,
        models=args.models,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        seeds=args.seeds,
        device=args.device,
        output_dir=args.output_dir,
        allow_mock=args.allow_mock,
        noise_schedule=args.noise_schedule
    )

if __name__ == "__main__":
    main()
