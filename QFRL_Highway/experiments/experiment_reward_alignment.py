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
    """Verifies TorchQuantum backend status."""
    try:
        import torchquantum as tq
        backend_path = str(getattr(tq, '__file__', ''))
        version = str(getattr(tq, '__version__', '0.3.0'))
        is_mock = "mock_torchquantum" in backend_path or ("QFRL_Highway" in backend_path and "mock" in backend_path)
        if is_mock:
            if not allow_mock:
                raise RuntimeError(f"Mock backend loaded at {backend_path}. Real backend required or use --allow_mock.")
            return tq, f"TorchQuantum {version} (Mock)", True
        return tq, f"TorchQuantum {version} (Real)", False
    except ImportError as e:
        if allow_mock:
            try:
                import mock_torchquantum as tq
                return tq, "TorchQuantum (Mock)", True
            except ImportError:
                pass
        raise RuntimeError(f"TorchQuantum not found: {e}. Pass --allow_mock for local testing.")

def run_reward_alignment_experiment(
    config_path: str = "configs/reward_alignment_config.yaml",
    models: list = ["classical"],
    reward_types: list = ["ratio", "negative_energy"],
    seeds: list = [42, 123, 2026],
    episodes: int = 15,
    steps_per_episode: int = 50,
    device: str = "cpu",
    output_dir: str = "results",
    output_csv: str = "reward_alignment_comparison.csv",
    allow_mock: bool = False,
    smoke_test: bool = False
):
    """
    Executes controlled reward-alignment experiment comparing:
      Condition A: 'ratio' (per-step completed bits / step energy)
      Condition B: 'negative_energy' (direct -step energy)
    Keeps model architecture, seeds, replay buffer, exploration noise, and steps matched.
    """
    if smoke_test:
        print("[SMOKE TEST] Running minimal smoke test (1 seed, 1 episode, 2 steps)...")
        seeds = [seeds[0]] if seeds else [42]
        episodes = 1
        steps_per_episode = 2

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, output_csv)
    csv_headers = [
        "run_id",
        "model_type",
        "reward_type",
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
        "frac_saturated_actions",
        "mean_tanh_derivative",
        "actor_grad_norm",
        "critic_grad_norm",
        "actor_trainable_params",
        "critic_trainable_params",
        "inference_time_ms",
        "duration_seconds",
        "backend_name"
    ]

    file_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, 'a' if file_exists else 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
    if not file_exists:
        writer.writeheader()

    buffer_size = config.get('rl', {}).get('replay_buffer_size', 6000)
    batch_size = config.get('rl', {}).get('batch_size', 32)
    warmup_steps = 32
    initial_noise = config.get('rl', {}).get('exploration_noise', 0.1)

    all_results = []

    print("\n" + "=" * 75)
    print("STAGE 9.3: CONTROLLED REWARD-ALIGNMENT EXPERIMENT")
    print(f"Models       : {models}")
    print(f"Reward Types : {reward_types}")
    print(f"Seeds        : {seeds}")
    print(f"Episodes     : {episodes}")
    print(f"Steps/Ep     : {steps_per_episode}")
    print(f"Device       : {device}")
    print(f"Output CSV   : {csv_path}")
    print("=" * 75 + "\n")

    for reward_type in reward_types:
        for model_name in models:
            print("\n" + "#" * 70)
            print(f"CONDITION: Reward='{reward_type}' | Model='{model_name}'")
            print("#" * 70)

            # Backend check
            if model_name == "quantum":
                _, curr_backend, _ = verify_quantum_backend(allow_mock=allow_mock)
            else:
                curr_backend = "Classical PyTorch MLP"

            for seed in seeds:
                print(f"\n--- [{model_name.upper()} | {reward_type}] Starting Seed {seed} ---")
                set_seed(seed)

                # Initialize environment with specific reward type
                run_config = copy.deepcopy(config)
                run_config['reward_type'] = reward_type
                run_config['device'] = device
                
                env = QFRLEnvironment()
                env.initialize_from_config(run_config)

                if model_name == "classical":
                    agent = ClassicalDDPGAgent(state_dim=150, action_dim=5, config=run_config)
                elif model_name == "quantum":
                    agent = DDPGAgent(config=run_config)
                else:
                    raise ValueError(f"Unknown model name: {model_name}")

                actor_params = count_trainable_parameters(agent.actor)
                critic_params = count_trainable_parameters(agent.critic)

                replay_buffer = ReplayBuffer(state_dim=150, action_dim=5, max_size=buffer_size)

                # Warm-up replay buffer
                w_state = env.reset()
                for _ in range(warmup_steps):
                    w_action = agent.select_action(w_state, add_noise=True)
                    w_next, w_r, w_d, w_info = env.step(w_action)
                    replay_buffer.add(w_state, w_action, w_r, w_next, w_d)
                    w_state = w_next
                    if w_d:
                        w_state = env.reset()

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
                    ep_actor_grads = []
                    ep_critic_grads = []
                    ep_sat_fracs = []
                    ep_tanh_derivs = []
                    inference_times = []

                    for step in range(steps_per_episode):
                        # Measure deterministic raw actor output to evaluate saturation
                        s_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                        with torch.no_grad():
                            raw_action = agent.actor(s_tensor).cpu().numpy()[0]
                        
                        sat_frac = float(np.mean(np.abs(raw_action) > 0.95))
                        tanh_deriv = float(np.mean(1.0 - (raw_action ** 2)))
                        ep_sat_fracs.append(sat_frac)
                        ep_tanh_derivs.append(tanh_deriv)

                        # Timed action selection with exploration noise
                        t0 = time.time()
                        action = agent.select_action(state, add_noise=True)
                        t_inf = (time.time() - t0) * 1000.0
                        inference_times.append(t_inf)

                        next_state, reward, done, info = env.step(action)
                        replay_buffer.add(state, action, reward, next_state, done)

                        al, cl = agent.train(replay_buffer, batch_size=batch_size)
                        if al is not None and cl is not None:
                            ep_actor_losses.append(al)
                            ep_critic_losses.append(cl)

                            # Record gradient norms
                            a_gn = sum(p.grad.norm().item()**2 for p in agent.actor.parameters() if p.grad is not None)**0.5
                            c_gn = sum(p.grad.norm().item()**2 for p in agent.critic.parameters() if p.grad is not None)**0.5
                            ep_actor_grads.append(a_gn)
                            ep_critic_grads.append(c_gn)

                        state = next_state
                        ep_reward += reward
                        ep_completed_bits += info.get('completed_bits', 0.0)
                        ep_total_energy += info.get('total_energy', 0.0)
                        ep_comp_ratios.append(info.get('task_completion_ratio', 1.0))

                        tx_time = info.get('transmission_time', 0.0)
                        cp_time = info.get('computation_time', 0.0)
                        ep_latencies.append(max(tx_time, cp_time))

                        if done:
                            break

                    ep_duration = time.time() - ep_start
                    mean_step_reward = ep_reward / max(step + 1, 1)
                    ee_mbits_per_joule = (ep_completed_bits / max(ep_total_energy, 1e-6)) / 1e6
                    avg_al = float(np.mean(ep_actor_losses)) if ep_actor_losses else float('nan')
                    avg_cl = float(np.mean(ep_critic_losses)) if ep_critic_losses else float('nan')
                    avg_a_gn = float(np.mean(ep_actor_grads)) if ep_actor_grads else float('nan')
                    avg_c_gn = float(np.mean(ep_critic_grads)) if ep_critic_grads else float('nan')
                    avg_sat = float(np.mean(ep_sat_fracs)) if ep_sat_fracs else float('nan')
                    avg_tanh_d = float(np.mean(ep_tanh_derivs)) if ep_tanh_derivs else float('nan')
                    mean_latency = float(np.mean(ep_latencies)) if ep_latencies else 0.0
                    mean_comp_ratio = float(np.mean(ep_comp_ratios)) if ep_comp_ratios else 0.0
                    mean_inf = float(np.mean(inference_times)) if inference_times else 0.0

                    unique_run_id = f"{model_name}_{reward_type}_s{seed}_ep{ep}_{int(time.time() * 1000) % 1000000}"

                    row = {
                        "run_id": unique_run_id,
                        "model_type": model_name,
                        "reward_type": reward_type,
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
                        "frac_saturated_actions": round(avg_sat, 4),
                        "mean_tanh_derivative": round(avg_tanh_d, 6),
                        "actor_grad_norm": round(avg_a_gn, 4) if np.isfinite(avg_a_gn) else "nan",
                        "critic_grad_norm": round(avg_c_gn, 4) if np.isfinite(avg_c_gn) else "nan",
                        "actor_trainable_params": actor_params,
                        "critic_trainable_params": critic_params,
                        "inference_time_ms": round(mean_inf, 3),
                        "duration_seconds": round(ep_duration, 2),
                        "backend_name": curr_backend
                    }

                    writer.writerow(row)
                    csv_file.flush()
                    all_results.append(row)

                    print(
                        f"  Ep {ep:02d}/{episodes:02d} | "
                        f"Rew: {row['episode_reward']:10.2f} | "
                        f"Energy: {row['total_energy']:8.2f} J | "
                        f"EE: {row['energy_efficiency_mbits_per_joule']:8.4f} Mb/J | "
                        f"Sat: {row['frac_saturated_actions']*100:5.1f}% | "
                        f"Tanh': {row['mean_tanh_derivative']:6.4f} | "
                        f"GradA: {row['actor_grad_norm']} | "
                        f"Time: {row['duration_seconds']:4.1f}s"
                    )

                    # Save intermediate checkpoints for non-smoke runs
                    if not smoke_test and ep in [5, 10, 15]:
                        ckpt_name = f"reward_exp_{model_name}_{reward_type}_seed{seed}_ep{ep}.pt"
                        agent.save_checkpoint(os.path.join(ckpt_dir, ckpt_name), ep, seed)

    csv_file.close()
    print("\n" + "=" * 75)
    print(f"REWARD ALIGNMENT EXPERIMENT COMPLETE. Saved to {csv_path}")
    print("=" * 75 + "\n")
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Controlled Reward-Alignment Experiment")
    parser.add_argument("--config", type=str, default="configs/reward_alignment_config.yaml")
    parser.add_argument("--models", nargs="+", default=["classical"], choices=["classical", "quantum"])
    parser.add_argument("--reward_types", nargs="+", default=["ratio", "negative_energy"], choices=["ratio", "negative_energy"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--steps_per_episode", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--output_csv", type=str, default="reward_alignment_comparison.csv")
    parser.add_argument("--allow_mock", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")

    args = parser.parse_args()

    run_reward_alignment_experiment(
        config_path=args.config,
        models=args.models,
        reward_types=args.reward_types,
        seeds=args.seeds,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        device=args.device,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        allow_mock=args.allow_mock,
        smoke_test=args.smoke_test
    )
