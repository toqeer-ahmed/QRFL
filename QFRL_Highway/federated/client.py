import torch
import numpy as np
from rl.ddpg import DDPGAgent
from rl.replay_buffer import ReplayBuffer
from environment.environment import QFRLEnvironment

class FederatedClient:
    def __init__(self, client_id, base_seed, config):
        self.client_id = client_id
        self.seed = base_seed + client_id
        self.config = config
        
        # IMPLEMENTATION CHOICE: Deterministic client seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Local Environment
        self.env = QFRLEnvironment()
        self.env.initialize_from_config(self.config)
        
        # Local RL Agent
        self.agent = DDPGAgent(state_dim=150, action_dim=5, config=self.config)
        self.actor = self.agent.actor
        self.critic = self.agent.critic
        self.target_actor = self.agent.actor_target
        self.target_critic = self.agent.critic_target
        self.actor_optimizer = self.agent.actor_optimizer
        self.critic_optimizer = self.agent.critic_optimizer
        
        # Local Replay Buffer
        self.replay_buffer = ReplayBuffer(state_dim=150, action_dim=5, max_size=self.config.get('replay_size', 6000))
        
        # State tracking
        self.current_state = self.env.reset()
        self.num_samples = 0

    def get_parameters(self):
        """Returns the local actor and critic state dicts without ephemeral state buffers."""
        return {
            'actor': {k: v.cpu().clone() for k, v in self.actor.state_dict().items() if not k.endswith('states')},
            'critic': {k: v.cpu().clone() for k, v in self.critic.state_dict().items()}
        }

    def set_parameters(self, global_params):
        """Updates local models with global parameters (both main and target)."""
        actor_sd = {k: v for k, v in global_params['actor'].items() if not k.endswith('states')}
        self.actor.load_state_dict(actor_sd, strict=False)
        self.critic.load_state_dict(global_params['critic'])
        self.target_actor.load_state_dict(actor_sd, strict=False)
        self.target_critic.load_state_dict(global_params['critic'])

    def collect_experience(self, steps=1):
        """Collects experience in the local environment."""
        episode_reward = 0.0
        for _ in range(steps):
            action = self.agent.select_action(self.current_state, add_noise=True)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.add(self.current_state, action, reward, next_state, done)
            self.current_state = next_state
            episode_reward += reward
            self.num_samples += 1
            if done:
                self.current_state = self.env.reset()
        return episode_reward

    def local_train(self, local_steps=1, batch_size=32):
        """Performs local training steps."""
        actor_losses = []
        critic_losses = []
        for _ in range(local_steps):
            al, cl = self.agent.train(self.replay_buffer, batch_size=batch_size)
            if al is not None and cl is not None:
                actor_losses.append(al)
                critic_losses.append(cl)
                
        mean_actor_loss = np.mean(actor_losses) if actor_losses else 0.0
        mean_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        return mean_actor_loss, mean_critic_loss

    def get_model_state(self):
        """Returns parameters and num samples for aggregation."""
        return {
            'params': self.get_parameters(),
            'num_samples': self.num_samples
        }
