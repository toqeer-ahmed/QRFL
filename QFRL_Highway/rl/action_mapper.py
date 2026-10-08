import numpy as np
import torch

class ActionMapper:
    """
    Converts raw normalized actor outputs [-1, 1] into physical environment actions.
    
    IMPLEMENTATION ASSUMPTION: 
    The paper specifies a 5D action vector but does not provide the mapping.
    This mapper defines:
    action[0] = offloading ratio
    action[1] = transmit power
    action[2] = IRS phase/control scalar
    action[3] = UAV vx
    action[4] = UAV vy
    """
    def __init__(self, config=None):
        self.config = config or {}
        
    def decode_action(self, action: np.ndarray) -> dict:
        """
        Input: action array of shape (5,) with elements in roughly [-1, 1]
        Output: Dictionary of physical variables.
        """
        # Ensure action is exactly bounded
        action = np.clip(action, -1.0, 1.0)
        
        # 1. Offloading Ratio: [-1, 1] -> [0, 1]
        offload_ratio = (action[0] + 1.0) / 2.0
        
        # 2. Transmit Power: [-1, 1] -> [0, 20] dBm
        # Using paper specified 20 dBm max
        tx_power = (action[1] + 1.0) / 2.0 * 20.0
        
        # 3. IRS Phase: [-1, 1] -> [-pi, pi]
        irs_phase = action[2] * np.pi
        
        # 4 & 5. UAV Velocity: [-1, 1] -> up to 10 m/s vector norm
        # IMPLEMENTATION FORMULATION: Projecting to satisfy UAV max speed (10 m/s)
        vx_raw = action[3] * 10.0
        vy_raw = action[4] * 10.0
        
        v_norm = np.sqrt(vx_raw**2 + vy_raw**2)
        if v_norm > 10.0:
            vx = (vx_raw / v_norm) * 10.0
            vy = (vy_raw / v_norm) * 10.0
        else:
            vx = vx_raw
            vy = vy_raw
            
        return {
            'offload_ratio': float(offload_ratio),
            'transmit_power': float(tx_power),
            'irs_phase': float(irs_phase),
            'uav_delta_x': float(vx),
            'uav_delta_y': float(vy)
        }
