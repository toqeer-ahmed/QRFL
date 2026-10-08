import numpy as np

class ReplayBuffer:
    """
    Reusable Replay Buffer for off-policy RL algorithms.
    Stores transitions: (state, action, reward, next_state, done).
    """
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 6000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        """Adds a transition to the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int = 32) -> dict:
        """Samples a random batch of transitions."""
        ind = np.random.randint(0, self.size, size=batch_size)
        return {
            'states': self.states[ind],
            'actions': self.actions[ind],
            'rewards': self.rewards[ind],
            'next_states': self.next_states[ind],
            'dones': self.dones[ind]
        }

    def __len__(self):
        return self.size
