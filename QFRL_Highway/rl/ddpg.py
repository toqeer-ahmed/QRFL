import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from quantum.qnn import QuantumActor

class Critic(nn.Module):
    """
    Classical Critic for QFRL.
    
    IMPLEMENTATION CHOICE:
    The exact critic architecture is not paper-specified. 
    We use a configurable modest MLP.
    Input: state (150) + action (5) = 155
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
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class DDPGAgent:
    """
    Single-Client DDPG Agent for QFRL.
    """
    def __init__(self, state_dim=150, action_dim=5, config=None):
        self.config = config or {}
        self.device = torch.device(self.config.get('device', 'cpu'))
        
        # PAPER-SPECIFIED hyperparameters
        self.gamma = self.config.get('gamma', 0.95)
        self.actor_lr = self.config.get('actor_lr', 0.001)
        self.critic_lr = self.config.get('critic_lr', 0.0025)
        
        # IMPLEMENTATION CHOICE hyperparameters
        self.tau = self.config.get('tau', 0.005)
        self.exploration_noise = self.config.get('exploration_noise', 0.1)
        
        # Initialize Actor and Critic
        self.actor = QuantumActor(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.critic = Critic(state_dim=state_dim, action_dim=action_dim, 
                             hidden_dim1=self.config.get('critic_hidden1', 64),
                             hidden_dim2=self.config.get('critic_hidden2', 64)).to(self.device)
                             
        self.actor_target = QuantumActor(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.critic_target = Critic(state_dim=state_dim, action_dim=action_dim, 
                                    hidden_dim1=self.config.get('critic_hidden1', 64),
                                    hidden_dim2=self.config.get('critic_hidden2', 64)).to(self.device)
                                    
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.critic_criterion = nn.MSELoss()

    def select_action(self, state: np.ndarray, add_noise=True, noise_scale=None) -> np.ndarray:
        """Selects a continuous action in [-1, 1]."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
            
        if add_noise:
            # IMPLEMENTATION CHOICE: Gaussian exploration noise
            scale = noise_scale if noise_scale is not None else self.exploration_noise
            noise = np.random.normal(0, scale, size=action.shape)
            action = action + noise
            
        return np.clip(action, -1.0, 1.0)
        
    def train(self, replay_buffer, batch_size=32):
        """Performs one step of DDPG training."""
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
        
        # Soft Target Updates (IMPLEMENTATION CHOICE)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return actor_loss.item(), critic_loss.item()
        
    def save_checkpoint(self, path: str, episode: int, seed: int):
        """Saves a checkpoint, excluding ephemeral simulation state buffers."""
        actor_sd = {k: v for k, v in self.actor.state_dict().items() if not k.endswith('states')}
        target_actor_sd = {k: v for k, v in self.actor_target.state_dict().items() if not k.endswith('states')}
        torch.save({
            'episode': episode,
            'seed': seed,
            'actor_state_dict': actor_sd,
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': target_actor_sd,
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config
        }, path)
        
    def load_checkpoint(self, path: str):
        """Loads a checkpoint, safely ignoring ephemeral quantum state buffers."""
        checkpoint = torch.load(path, map_location=self.device)
        actor_sd = {k: v for k, v in checkpoint['actor_state_dict'].items() if not k.endswith('states')}
        target_actor_sd = {k: v for k, v in checkpoint['actor_target_state_dict'].items() if not k.endswith('states')}
        self.actor.load_state_dict(actor_sd, strict=False)
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(target_actor_sd, strict=False)
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        return checkpoint['episode'], checkpoint['seed'], checkpoint['config']
