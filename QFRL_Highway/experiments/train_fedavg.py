import argparse
import yaml
import os
import csv
import torch
import numpy as np

from federated.client import FederatedClient
from federated.server import FederatedServer

def main():
    parser = argparse.ArgumentParser(description="FedAvg Smoke Test for QFRL")
    parser.add_argument("--clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=2, help="Number of federated rounds")
    parser.add_argument("--local_steps", type=int, default=2, help="Number of local training steps")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")
    
    args = parser.parse_args()
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Load base config
    config_path = "configs/paper_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    config['device'] = args.device
    
    print(f"Initializing FedAvg with {args.clients} clients for {args.rounds} rounds.")
    
    server = FederatedServer(num_clients=args.clients)
    clients = [FederatedClient(client_id=i, base_seed=args.seed, config=config) for i in range(args.clients)]
    
    # Initialize global model
    server.initialize_global_model(clients[0].get_parameters())
    
    csv_file = "results/fedavg_smoke.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'client', 'actor_loss', 'critic_loss', 'num_samples', 'global_actor_norm', 'global_critic_norm'])
    
    for round_num in range(1, args.rounds + 1):
        print(f"\n--- Round {round_num} ---")
        
        # Broadcast global parameters
        global_params = server.get_global_parameters()
        for c in clients:
            c.set_parameters(global_params)
            
        client_states = []
        round_results = []
        
        # Local updates
        for c in clients:
            c.collect_experience(steps=10) # Collect a small batch of transitions
            actor_loss, critic_loss = c.local_train(local_steps=args.local_steps, batch_size=32)
            
            c_state = c.get_model_state()
            client_states.append(c_state)
            
            print(f"Client {c.client_id} | Samples: {c_state['num_samples']} | Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f}")
            
            round_results.append({
                'client': c.client_id,
                'actor_loss': actor_loss,
                'critic_loss': critic_loss,
                'num_samples': c_state['num_samples']
            })
            
        # Aggregation
        print(f"Aggregating {len(clients)} client updates...")
        global_params = server.aggregate(client_states)
        
        # Calculate norms
        actor_norm = sum(torch.norm(v).item() for v in global_params['actor'].values())
        critic_norm = sum(torch.norm(v).item() for v in global_params['critic'].values())
        
        print(f"Global Actor Norm: {actor_norm:.4f} | Global Critic Norm: {critic_norm:.4f}")
        
        # Save to CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            for res in round_results:
                writer.writerow([round_num, res['client'], res['actor_loss'], res['critic_loss'], res['num_samples'], actor_norm, critic_norm])
                
        # Save checkpoint
        server.save_checkpoint(f"checkpoints/fedavg_round_{round_num}.pt", round_num, args.seed, config)

    print("\nFedAvg Smoke Test Completed Successfully.")

if __name__ == "__main__":
    main()
