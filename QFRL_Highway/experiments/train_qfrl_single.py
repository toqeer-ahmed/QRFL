import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import argparse
import torch
import torchquantum as tq
import numpy as np

from rl.ddpg import DDPGAgent
from rl.replay_buffer import ReplayBuffer
from environment.environment import QFRLEnvironment

def get_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Single-Client QFRL/DDPG Test Script")
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes (small default for testing)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--checkpoint_dir', type=str, default='results/checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--config', type=str, default='configs/paper_config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Reproducibility
    print(f"Seed: {args.seed}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device Support
    print(f"Device: {args.device}")
    print(f"Torch version: {torch.__version__}")
    print(f"TorchQuantum version: {tq.__version__}")
    print(f"Number of QNN qubits: 4")
    print(f"Number of measurement shots: 2048")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load config and environment
    config = get_config(args.config)
    config['device'] = args.device
    
    env = QFRLEnvironment()
    env.initialize_from_config(config)
    
    # Initialize Agent and Buffer
    agent = DDPGAgent(config=config)
    # The paper uses 6000 buffer size
    buffer_size = config.get('rl', {}).get('replay_buffer_size', 6000)
    batch_size = config.get('rl', {}).get('batch_size', 32)
    
    replay_buffer = ReplayBuffer(state_dim=150, action_dim=5, max_size=buffer_size)
    
    # Training Loop
    for ep in range(1, args.episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_bits = 0
        episode_energy = 0
        actor_losses = []
        critic_losses = []
        
        done = False
        step = 0
        
        while not done:
            # For testing, we just use the max_steps from env config (or break early if testing is meant to be very short)
            action = agent.select_action(state, add_noise=True)
            next_state, reward, done, info = env.step(action)
            
            replay_buffer.add(state, action, reward, next_state, done)
            
            # Train step
            al, cl = agent.train(replay_buffer, batch_size=batch_size)
            if al is not None:
                actor_losses.append(al)
                critic_losses.append(cl)
                
            state = next_state
            episode_reward += reward
            episode_bits += info.get('completed_bits', 0)
            episode_energy += info.get('total_energy', 0)
            step += 1
            
            # Very short step limit for smoke testing if episodes are few
            if args.episodes <= 10 and step >= 20: 
                break

        # Logging for short test run
        avg_al = np.mean(actor_losses) if actor_losses else 0.0
        avg_cl = np.mean(critic_losses) if critic_losses else 0.0
        print(f"Episode: {ep} | Reward: {episode_reward:.4f} | Bits: {episode_bits:.2e} | "
              f"Energy: {episode_energy:.4f} | ALoss: {avg_al:.4f} | CLoss: {avg_cl:.4f} | "
              f"Buffer: {len(replay_buffer)}")
              
        # Debugging prints
        print(f"  [Debug] State Shape: {state.shape}, Action Shape: {action.shape}")
        
        phys_action = env.action_mapper.decode_action(action)
        print(f"  [Debug] Physical Action: {phys_action}")
        
        vx, vy = phys_action['uav_delta_x'], phys_action['uav_delta_y']
        print(f"  [Debug] UAV Velocity Norm: {np.sqrt(vx**2 + vy**2):.4f}")
        
        phases = env.irs_controller.generate_phase_vector(phys_action['irs_phase'])
        print(f"  [Debug] IRS Phase (element 0): {phases[0]:.4f}")
        
        # Checkpoint
        if ep == args.episodes:
            ckpt_path = os.path.join(args.checkpoint_dir, f"qfrl_ckpt_ep{ep}.pt")
            agent.save_checkpoint(ckpt_path, episode=ep, seed=args.seed)
            print(f"Checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    main()
