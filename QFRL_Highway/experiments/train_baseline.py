import os
import yaml
import sys

# Ensure relative imports work when executed from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.environment import QFRLEnvironment
from rl.agent import PlaceholderAgent
from rl.replay_buffer import ReplayBuffer

def main():
    """
    TRAINING SCRIPT SKELETON
    Designed to be a baseline for future Kaggle runs.
    """
    # 1. Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'paper_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # RL Params
    rl_cfg = config.get('rl', {})
    state_dim = rl_cfg.get('state_dim', 150)
    action_dim = rl_cfg.get('action_dim', 5)
    buffer_size = rl_cfg.get('replay_buffer_size', 6000)
    batch_size = rl_cfg.get('batch_size', 32)
    # The actual paper specifies 8000 training epochs. 
    # For this skeleton/test script, we only run a small number.
    # A true long run is strictly forbidden in this stage!
    training_epochs = 2  # Hardcoded small limit for this integration stage
    
    print(f"Loaded config: State Dim: {state_dim}, Action Dim: {action_dim}")

    # 2. Create environment
    env = QFRLEnvironment()
    env.initialize_from_config(config)

    # 3. Initialize agent placeholder & buffer
    agent = PlaceholderAgent(state_dim, action_dim)
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=buffer_size)

    # Directories for saving
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    checkpoints_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    print("Starting Training Skeleton Loop...")
    
    for epoch in range(training_epochs):
        # 4. Reset environment
        state = env.reset()
        epoch_reward = 0
        done = False
        
        # 5. Execute episodes
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # 6 & 7. Collect and add transitions
            replay_buffer.add(state, action, reward, next_state, done)
            
            # Train step (placeholder)
            agent.train(replay_buffer, batch_size=batch_size)
            
            state = next_state
            epoch_reward += reward

        # 8. Print/Log metrics
        print(f"Epoch {epoch+1}/{training_epochs} - Reward: {epoch_reward:.4f} - Buffer Size: {len(replay_buffer)}")
        
        # 9 & 10. Save results & checkpoint support
        if (epoch + 1) % 1 == 0:
            ckpt_path = os.path.join(checkpoints_dir, f'ckpt_epoch_{epoch+1}.pt')
            # fake save
            with open(ckpt_path, 'w') as f:
                f.write(f"Checkpoint for epoch {epoch+1}")
            print(f"Saved checkpoint: {ckpt_path}")

    print("Training Skeleton Execution Completed.")

if __name__ == '__main__':
    main()
