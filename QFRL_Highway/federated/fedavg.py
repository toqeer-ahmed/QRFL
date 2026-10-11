import torch

def fedavg(client_states):
    """
    Performs standard sample-weighted Federated Averaging.
    client_states: list of dicts {'params': {'actor': state_dict, 'critic': state_dict}, 'num_samples': int}
    """
    total_samples = sum(cs['num_samples'] for cs in client_states)
    
    if total_samples == 0:
        # If no samples, return uniform average
        weights = [1.0 / len(client_states) for _ in client_states]
    else:
        weights = [cs['num_samples'] / total_samples for cs in client_states]
    
    # Check model parameter consistency
    base_actor_keys = list(client_states[0]['params']['actor'].keys())
    base_critic_keys = list(client_states[0]['params']['critic'].keys())
    
    for i, cs in enumerate(client_states):
        actor_keys = list(cs['params']['actor'].keys())
        critic_keys = list(cs['params']['critic'].keys())
        if actor_keys != base_actor_keys:
            raise ValueError(f"Actor parameter mismatch in client {i}")
        if critic_keys != base_critic_keys:
            raise ValueError(f"Critic parameter mismatch in client {i}")
            
        # Check shapes
        for k in base_actor_keys:
            if cs['params']['actor'][k].shape != client_states[0]['params']['actor'][k].shape:
                raise ValueError(f"Actor shape mismatch for {k} in client {i}")
        for k in base_critic_keys:
            if cs['params']['critic'][k].shape != client_states[0]['params']['critic'][k].shape:
                raise ValueError(f"Critic shape mismatch for {k} in client {i}")
                
    aggregated_actor = {}
    aggregated_critic = {}
    
    for k in base_actor_keys:
        sample_tensor = client_states[0]['params']['actor'][k]
        if sample_tensor.is_complex():
            weighted_sum = torch.zeros_like(sample_tensor)
            for i, cs in enumerate(client_states):
                weighted_sum += cs['params']['actor'][k] * weights[i]
            aggregated_actor[k] = weighted_sum
        else:
            weighted_sum = torch.zeros_like(sample_tensor, dtype=torch.float32)
            for i, cs in enumerate(client_states):
                weighted_sum += cs['params']['actor'][k].float() * weights[i]
            aggregated_actor[k] = weighted_sum.to(sample_tensor.dtype)
        
    for k in base_critic_keys:
        sample_tensor = client_states[0]['params']['critic'][k]
        weighted_sum = torch.zeros_like(sample_tensor, dtype=torch.float32)
        for i, cs in enumerate(client_states):
            weighted_sum += cs['params']['critic'][k].float() * weights[i]
        aggregated_critic[k] = weighted_sum.to(sample_tensor.dtype)
        
    return {
        'actor': aggregated_actor,
        'critic': aggregated_critic
    }
