import os
import sys
import yaml
import json
import random
import numpy as np

# In a real PyTorch run, this would be: import torch

# Ensure relative imports work when executed from root or Kaggle env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.environment import QFRLEnvironment
from rl.agent import PlaceholderAgent
from rl.replay_buffer import ReplayBuffer

def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    print(f"Random seeds set to {seed}")

def check_gpu():
    """Detect and print GPU information."""
    print("=== Hardware Information ===")
    # try:
    #     if torch.cuda.is_available():
    #         print(f"CUDA is available! Device: {torch.cuda.get_device_name(0)}")
    #     else:
    #         print("CUDA is NOT available. Running on CPU.")
    # except Exception as e:
    #     print(f"Could not verify CUDA: {e}")
    print("PyTorch not yet imported for this stage, but GPU check placeholder is ready.")
    print("============================")

def main():
    """
    KAGGLE ENTRY POINT
    Future main training script for execution on Kaggle GPUs.
    """
    print("Starting Kaggle QFRL Training Pipeline Initialization...")
    set_seeds(42)
    check_gpu()

    # Use relative paths for Kaggle compatibility
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(base_dir, 'configs', 'paper_config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # RL Params
    rl_cfg = config.get('rl', {})
    state_dim = rl_cfg.get('state_dim', 150)
    action_dim = rl_cfg.get('action_dim', 5)
    buffer_size = rl_cfg.get('replay_buffer_size', 6000)
    
    print(f"Configuration Loaded: State({state_dim}), Action({action_dim})")

    # Directories
    results_dir = os.path.join(base_dir, 'results')
    checkpoints_dir = os.path.join(base_dir, 'checkpoints')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Setup 
    env = QFRLEnvironment()
    env.initialize_from_config(config)
    agent = PlaceholderAgent(state_dim, action_dim)
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=buffer_size)

    # Save initial experiment metadata
    metadata = {
        'config': config,
        'status': 'initialized',
        'kaggle_ready': True
    }
    with open(os.path.join(results_dir, 'experiment_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Kaggle Setup Complete. Ready for future QFRL training epochs.")
    print(f"Results will be saved to: {results_dir}")
    print(f"Checkpoints will be saved to: {checkpoints_dir}")

if __name__ == '__main__':
    main()
