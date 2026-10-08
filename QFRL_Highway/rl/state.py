import numpy as np

class StateRepresentation:
    """
    Handles the conversion of the environment state into a 150-dimensional 
    vector as specified by the paper.
    
    Paper reports state dimension = 150, but the exact feature ordering is 
    not explicitly specified. 
    
    Implementation Choice (Feature Allocation):
    - 2 dims: UAV position (x, y)
    - 2 dims: RSU position (x, y)
    - For up to 30 vehicles (120 dims total, 4 per vehicle):
        - Vehicle X position
        - Vehicle Y position
        - Task input bits
        - Vehicle CPU frequency
    - Remaining 26 dims:
        - UAV CPU frequency (1)
        - RSU CPU frequency (1)
        - IRS base position (3)
        - Channel/Environment summary stats (21 zeros/padding for future-proofing)
    
    Total: 2 + 2 + 120 + 26 = 150.
    """
    def __init__(self, state_dim: int = 150):
        self.state_dim = state_dim

    def normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Min-max normalization."""
        if max_val == min_val:
            return 0.0
        # Clip to ensure valid range
        val = max(min_val, min(max_val, value))
        return (val - min_val) / (max_val - min_val)

    def extract_state(self, env) -> np.ndarray:
        """
        Extracts the state from the environment and normalizes it.
        """
        state = np.zeros(self.state_dim, dtype=np.float32)
        idx = 0

        # UAV (2 dims)
        if env.uav:
            state[idx] = self.normalize(env.uav.x, -1000, 1000)
            state[idx+1] = self.normalize(env.uav.y, -1000, 1000)
        idx += 2

        # RSU (2 dims)
        if env.rsus and len(env.rsus) > 0:
            state[idx] = self.normalize(env.rsus[0].x, -1000, 1000)
            state[idx+1] = self.normalize(env.rsus[0].y, -1000, 1000)
        idx += 2

        # Vehicles (up to 30 * 4 = 120 dims)
        max_vehicles = 30
        for i in range(max_vehicles):
            if i < len(env.vehicles):
                v = env.vehicles[i]
                state[idx] = self.normalize(v.x, -1000, 1000)
                state[idx+1] = self.normalize(v.y, -1000, 1000)
                state[idx+3] = self.normalize(v.cpu_frequency, 1.0, 5.0)
                
                # Match task
                task = next((t for t in env.tasks if t.originating_vehicle_id == v.vehicle_id), None)
                if task:
                    state[idx+2] = self.normalize(task.input_bits, 1e5, 1e7)
                else:
                    state[idx+2] = 0.0
            idx += 4

        # Miscellaneous/Padding (26 dims)
        if env.uav:
            state[idx] = self.normalize(env.uav.cpu_frequency, 1.0, 15.0)
        idx += 1
        
        if env.rsus and len(env.rsus) > 0:
            state[idx] = self.normalize(env.rsus[0].cpu_frequency, 1.0, 15.0)
        idx += 1
        
        if env.irs:
            state[idx] = self.normalize(env.irs.base_x, -1000, 1000)
            state[idx+1] = self.normalize(env.irs.base_y, -1000, 1000)
            state[idx+2] = self.normalize(env.irs.base_z, 0, 100)
        idx += 3

        # Remaining 21 dims are left as 0 (Channel/Environment summary stats)
        # Pad with 0s inherently done by np.zeros
        
        return state
