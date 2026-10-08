import numpy as np

class PlaceholderAgent:
    """
    Placeholder/Random-Action Agent strictly for environment integration testing.
    DO NOT call this a QFRL result. The actual QFRL agent will be plugged in later.
    """
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Returns a random valid normalized action in the range [-1.0, 1.0].
        """
        return np.random.uniform(-1.0, 1.0, size=(self.action_dim,))

    def train(self, replay_buffer, batch_size: int = 32):
        """
        Placeholder for the training step.
        """
        if len(replay_buffer) < batch_size:
            return None
        
        batch = replay_buffer.sample(batch_size)
        # Fake training logic...
        loss = np.random.random()
        return {'loss': loss}
