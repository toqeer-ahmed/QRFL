import numpy as np

class ActionRepresentation:
    """
    Handles the 5-dimensional action vector from the RL agent and maps it to
    physical environment variables.
    
    Paper reports action dimension = 5.
    The optimization variables are related to:
    - task offloading
    - transmit power
    - IRS phase configuration
    - UAV trajectory
    
    Implementation Choice (Action Mapping):
    Since the paper does not explicitly map the 5 continuous dimensions, we map them as follows:
    - action[0]: Task offloading ratio (0.0 to 1.0)
    - action[1]: Transmit power (normalized to [0, max_transmit_power])
    - action[2]: IRS phase configuration (single continuous parameter controlling phase shifts, e.g., -pi to pi)
    - action[3]: UAV delta x (trajectory component X)
    - action[4]: UAV delta y (trajectory component Y)
    
    The inputs are expected to be in the range [-1.0, 1.0] from a typical bounded actor (e.g., Tanh).
    """
    def __init__(self, action_dim: int = 5):
        self.action_dim = action_dim

    def decode_action(self, action: np.ndarray, env_config: dict) -> dict:
        """
        Decodes the normalized RL action [-1, 1] to physical constraints.
        """
        # Clip action to [-1, 1] to prevent instability
        action = np.clip(action, -1.0, 1.0)
        
        # 1. Offloading Ratio [0, 1]
        offload_ratio = (action[0] + 1.0) / 2.0
        
        # 2. Transmit power (e.g., 0 to max_power)
        # Using 20 dBm max from config, approx 100mW.
        max_power = env_config.get('vehicles', {}).get('max_transmit_power_dbm', 20.0)
        # Linear scaling to dBm (or could scale in Watts)
        transmit_power = (action[1] + 1.0) / 2.0 * max_power
        
        # 3. IRS phase configuration [-pi, pi]
        irs_phase = action[2] * np.pi
        
        # 4 & 5. UAV trajectory (velocity based)
        uav_max_speed = env_config.get('uav', {}).get('max_velocity_kmh', 36.0)
        # Convert km/h to m/s
        uav_max_speed_ms = uav_max_speed * (1000.0 / 3600.0)
        
        uav_delta_x = action[3] * uav_max_speed_ms
        uav_delta_y = action[4] * uav_max_speed_ms
        
        return {
            'offload_ratio': offload_ratio,
            'transmit_power': transmit_power,
            'irs_phase': irs_phase,
            'uav_delta_x': uav_delta_x,
            'uav_delta_y': uav_delta_y
        }
