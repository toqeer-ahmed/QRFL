import torch
import os
from federated.fedavg import fedavg

class FederatedServer:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.global_parameters = None

    def initialize_global_model(self, initial_params):
        """Sets the initial global model parameters."""
        self.global_parameters = initial_params

    def aggregate(self, client_states):
        """Aggregates client states and updates the global model."""
        self.global_parameters = fedavg(client_states)
        return self.global_parameters

    def get_global_parameters(self):
        return self.global_parameters
        
    def save_checkpoint(self, path, round_num, seed, config):
        torch.save({
            'global_actor': self.global_parameters['actor'],
            'global_critic': self.global_parameters['critic'],
            'round': round_num,
            'clients': self.num_clients,
            'seed': seed,
            'config': config
        }, path)
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.global_parameters = {
            'actor': checkpoint['global_actor'],
            'critic': checkpoint['global_critic']
        }
        return checkpoint['round'], checkpoint['seed'], checkpoint['clients'], checkpoint['config']
