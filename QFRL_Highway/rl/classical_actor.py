import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Critic(nn.Module):
    """
    Classical MLP Critic for QFRL.
    Matches the exact critic architecture used in the quantum baseline:
    Input: state (150) + action (5) = 155
    Hidden: 64 -> ReLU -> 64 -> ReLU -> 1
    Output: Q-value (1)
    """
    def __init__(self, state_dim=150, action_dim=5, hidden_dim1=64, hidden_dim2=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class ClassicalActor(nn.Module):
    """
    Classical MLP Actor for QFRL Benchmark.
    
    Architecture:
    Maps 150-dimensional environment state -> Hidden Layers -> 5-dimensional continuous action in [-1, 1].
    Uses Tanh activation on the final layer to guarantee outputs are strictly within [-1.0, 1.0].
    
    Default Architecture:
    Linear(150, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 5) -> Tanh
    """
    def __init__(self, state_dim=150, action_dim=5, hidden_dims=(64, 64)):
        super(ClassicalActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = list(hidden_dims)
        
        layers = []
        prev_dim = state_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            state (torch.Tensor): Tensor of shape (batch_size, state_dim) or (state_dim,)
        Returns:
            torch.Tensor: Action tensor of shape (batch_size, action_dim) bounded in [-1.0, 1.0]
        """
        return self.network(state)

class ClassicalDDPGAgent:
    """
    Classical DDPG Agent using ClassicalActor and Classical Critic.
    Maintains 100% interface compatibility with DDPGAgent while ensuring
    zero quantum dependencies or quantum execution overhead.
    """
    def __init__(self, state_dim=150, action_dim=5, config=None):
        self.config = config or {}
        self.device = torch.device(self.config.get('device', 'cpu'))
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Hyperparameters (matched to paper and quantum baseline)
        self.gamma = self.config.get('gamma', 0.95)
        self.actor_lr = self.config.get('actor_lr', 0.001)
        self.critic_lr = self.config.get('critic_lr', 0.0025)
        self.tau = self.config.get('tau', 0.005)
        self.exploration_noise = self.config.get('exploration_noise', 0.1)
        
        hidden_dims = self.config.get('classical_actor_hidden', (64, 64))
        
        # Initialize Classical Actor & Targets
        self.actor = ClassicalActor(state_dim=state_dim, action_dim=action_dim, hidden_dims=hidden_dims).to(self.device)
        self.actor_target = ClassicalActor(state_dim=state_dim, action_dim=action_dim, hidden_dims=hidden_dims).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Initialize Classical Critic & Targets (identical to quantum baseline)
        critic_h1 = self.config.get('critic_hidden1', 64)
        critic_h2 = self.config.get('critic_hidden2', 64)
        self.critic = Critic(state_dim=state_dim, action_dim=action_dim, hidden_dim1=critic_h1, hidden_dim2=critic_h2).to(self.device)
        self.critic_target = Critic(state_dim=state_dim, action_dim=action_dim, hidden_dim1=critic_h1, hidden_dim2=critic_h2).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers (identical settings to quantum baseline)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.critic_criterion = nn.MSELoss()
        
    def select_action(self, state: np.ndarray, add_noise=True, noise_scale=None) -> np.ndarray:
        """
        Selects a continuous action in [-1.0, 1.0].
        Args:
            state (np.ndarray): Environment state of shape (150,)
            add_noise (bool): Whether to add Gaussian exploration noise
            noise_scale (float, optional): Custom noise standard deviation
        Returns:
            np.ndarray: Action vector of shape (5,) in [-1.0, 1.0]
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
            
        if add_noise:
            scale = noise_scale if noise_scale is not None else self.exploration_noise
            noise = np.random.normal(0, scale, size=action.shape)
            action = action + noise
            
        return np.clip(action, -1.0, 1.0)
        
    def train(self, replay_buffer, batch_size=32):
        """
        Performs one gradient update step of DDPG.
        Returns:
            tuple: (actor_loss, critic_loss) or (None, None) if insufficient samples
        """
        if len(replay_buffer) < batch_size:
            return None, None
            
        batch = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(batch['states']).to(self.device)
        action = torch.FloatTensor(batch['actions']).to(self.device)
        reward = torch.FloatTensor(batch['rewards']).to(self.device)
        next_state = torch.FloatTensor(batch['next_states']).to(self.device)
        done = torch.FloatTensor(batch['dones']).to(self.device)
        
        # Critic Update
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * target_q
            
        current_q = self.critic(state, action)
        critic_loss = self.critic_criterion(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor Update
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft Target Updates
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return actor_loss.item(), critic_loss.item()
        
    def save_checkpoint(self, path: str, episode: int, seed: int):
        """Saves classical agent checkpoint."""
        torch.save({
            'episode': episode,
            'seed': seed,
            'actor_type': 'classical',
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config
        }, path)
        
    def load_checkpoint(self, path: str):
        """Loads classical agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        return checkpoint['episode'], checkpoint['seed'], checkpoint['config']
