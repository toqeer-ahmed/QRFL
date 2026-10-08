class RewardFunction:
    """
    Calculates the energy efficiency reward as specified in the paper.
    
    Objective: ENERGY EFFICIENCY
    Conceptually defined as: completed task bits / total energy consumption
    """
    def __init__(self):
        pass

    def calculate_reward(self, env_state_info: dict) -> tuple:
        """
        Calculates reward and diagnostic info based on physical quantities.
        
        Args:
            env_state_info: Dictionary containing physical measurements from the env step.
            
        Returns:
            reward (float): The main energy efficiency reward.
            info (dict): Diagnostic information.
        """
        # Retrieve values, default to small non-zero energy to avoid division by zero
        completed_bits = env_state_info.get('completed_bits', 0.0)
        total_energy = env_state_info.get('total_energy', 1e-6) 
        
        # IMPLEMENTATION CHOICE: Epsilon guard for numerical stability
        epsilon = 1e-6
        # Guard against zero/negative energy in calculation
        total_energy = max(total_energy, epsilon)
            
        energy_efficiency = completed_bits / total_energy
        
        # Reward is directly the energy efficiency (or scaled if needed)
        reward = energy_efficiency
        
        info = {
            'completed_bits': completed_bits,
            'total_energy': total_energy,
            'energy_efficiency': energy_efficiency,
            'task_completion_ratio': env_state_info.get('task_completion_ratio', 0.0),
            'transmission_time': env_state_info.get('transmission_time', 0.0),
            'computation_time': env_state_info.get('computation_time', 0.0),
            'communication_energy': env_state_info.get('communication_energy', 0.0),
            'computation_energy': env_state_info.get('computation_energy', 0.0),
            'local_computation_energy': env_state_info.get('local_computation_energy', 0.0),
            'rsu_computation_energy': env_state_info.get('rsu_computation_energy', 0.0),
            'uav_computation_energy': env_state_info.get('uav_computation_energy', 0.0)
        }
        
        return reward, info
