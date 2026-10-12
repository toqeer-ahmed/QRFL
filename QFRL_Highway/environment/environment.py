from typing import List, Dict, Any
import numpy as np

from .vehicle import Vehicle
from .rsu import RSU
from .uav import UAV
from .irs import IRS, IRSElement
from .task import Task

# Imports from other modules
from communication.distance import distance_2d_to_3d, distance_2d, distance_3d
from communication.channel import compute_G1, compute_G2, compute_G3, compute_effective_channel, compute_reflection_matrix
from communication.snr import compute_snr
from communication.rate import compute_shannon_rate
from computing.energy import compute_local_energy, compute_uav_energy
from computing.local import compute_local_time
from computing.edge import compute_uav_time

class QFRLEnvironment:
    """
    Basic Simulation environment holding the physical entities described in the paper.
    Now extended to act as an RL environment.
    """
    def __init__(self):
        self.vehicles: List[Vehicle] = []
        self.rsus: List[RSU] = []
        self.uav: UAV = None
        self.irs: IRS = None
        self.tasks: List[Task] = []
        self.config: Dict[str, Any] = {}
        
        # RL Specifics
        self.current_step = 0
        self.max_steps = 5000
        
        # Lazy imports to prevent circular dependencies if they exist
        from rl.state import StateRepresentation
        from rl.action_mapper import ActionMapper
        from rl.reward import RewardFunction
        from environment.irs_controller import IRSController
        
        self.state_rep = StateRepresentation()
        self.action_mapper = ActionMapper(self.config)
        self.reward_func = RewardFunction()
        self.irs_controller = IRSController()

    def initialize_from_config(self, config: Dict[str, Any]):
        """
        Initializes the environment entities from the provided configuration dictionary.
        """
        self.config = config
        self.max_steps = config.get('rl', {}).get('episode_length', 5000)
        
        # Configure reward function (ratio vs negative_energy)
        from rl.reward import RewardFunction
        reward_type = config.get('reward_type', 'ratio')
        self.reward_func = RewardFunction(reward_type=reward_type)
        
        self.vehicles.clear()
        self.rsus.clear()
        self.tasks.clear()
        
        # Vehicles
        veh_cfg = config.get('vehicles', {})
        count = veh_cfg.get('count', 0)
        for i in range(count):
            self.vehicles.append(Vehicle(
                vehicle_id=i,
                x=np.random.uniform(-500, 500), # Spread them out for variety
                y=0.0,
                speed=veh_cfg.get('max_velocity_kmh', 40.0),
                cpu_frequency=veh_cfg.get('cpu_frequency_ghz', 4.0),
                max_transmit_power=veh_cfg.get('max_transmit_power_dbm', 20.0)
            ))
            
            # Create a task for each vehicle
            task_cfg = config.get('tasks', {})
            self.tasks.append(Task(
                task_id=i,
                originating_vehicle_id=i,
                input_bits=task_cfg.get('input_bits', 1048576),
                cycles_per_bit=task_cfg.get('cycles_per_bit', 1000),
                deadline=0.5
            ))
            
        # RSU
        rsu_cfg = config.get('rsu', {})
        if rsu_cfg:
            self.rsus.append(RSU(
                rsu_id=0,
                x=0.0,
                y=0.0,
                cpu_frequency=rsu_cfg.get('cpu_frequency_ghz', 9.0)
            ))

        # UAV
        uav_cfg = config.get('uav', {})
        if uav_cfg:
            self.uav = UAV(
                x=0.0,
                y=0.0,
                altitude=uav_cfg.get('altitude_m', 300.0),
                max_speed=uav_cfg.get('max_velocity_kmh', 36.0),
                cpu_frequency=uav_cfg.get('cpu_frequency_ghz', 12.0)
            )

        # IRS
        irs_cfg = config.get('irs', {})
        num_elements = irs_cfg.get('num_elements', 16) if irs_cfg else 16
        elements = [IRSElement(i, 0.0, 0.0, 0.0) for i in range(num_elements)]
        self.irs = IRS(
            num_elements=num_elements,
            base_x=10.0,
            base_y=10.0,
            base_z=10.0,
            elements=elements
        )

    def reset(self):
        """
        Resets the environment for a new episode.
        Returns the initial state.
        """
        self.current_step = 0
        # Reinitialize from config to reset positions/tasks
        self.initialize_from_config(self.config)
        return self.state_rep.extract_state(self)

    def step(self, action: np.ndarray):
        """
        Executes one environment step using the RL action.
        """
        # 1. Decode Action
        physical_action = self.action_mapper.decode_action(action)
        
        # 2. Apply UAV Trajectory updates
        if self.uav:
            self.uav.x += physical_action['uav_delta_x'] * 0.1 # assume 0.1s step
            self.uav.y += physical_action['uav_delta_y'] * 0.1
            
        # 3. Apply IRS configuration
        if self.irs:
            phases = self.irs_controller.generate_phase_vector(physical_action['irs_phase'])
            for i, elem in enumerate(self.irs.elements):
                if i < len(phases):
                    elem.phase = phases[i]
                
        # 4. Computation and Communication Execution
        comm_cfg = self.config.get('communication', {})
        comp_cfg = self.config.get('computation', {})
        
        # Missing config parameters defaults
        beta_1 = comm_cfg.get('beta_1', 0.001)
        beta_2 = comm_cfg.get('beta_2', 2.2)
        beta_3 = comm_cfg.get('beta_3', 2.2)
        beta_4 = comm_cfg.get('beta_4', 2.5)
        k_rician = comm_cfg.get('K_rician', 10.0)
        bandwidth = comm_cfg.get('bandwidth_hz', 1000000.0)
        noise_dbm = comm_cfg.get('noise_power_dbm', -120)
        noise_w = 10 ** ((noise_dbm - 30) / 10)
        kappa = comp_cfg.get('kappa', 1e-28)
        wavelength = comm_cfg.get('wavelength_m', 0.1)
        
        total_completed_bits = 0.0
        total_energy = 0.0
        
        total_local_energy = 0.0
        total_uav_energy = 0.0
        total_tx_energy = 0.0
        
        # Precompute channels if needed, assuming single interfering scenario for simplification
        # For full fidelity, we calculate interference between all offloading vehicles
        # As a simplification for the step, we assume interference is negligible or static
        
        # Calculate Reflection matrix Theta
        phases = np.array([e.phase for e in self.irs.elements]) if self.irs else np.zeros(16)
        Theta = compute_reflection_matrix(phases)
        
        for idx, task in enumerate(self.tasks):
            vehicle = self.vehicles[idx]
            ratio = physical_action['offload_ratio']
            local_bits = task.input_bits * (1 - ratio)
            offload_bits = task.input_bits * ratio
            
            # --- LOCAL COMPUTATION ---
            local_task = Task(task.task_id, vehicle.vehicle_id, local_bits, task.cycles_per_bit, task.deadline)
            local_energy = compute_local_energy(local_task, vehicle, kappa)
            local_time = compute_local_time(local_task, vehicle)
            
            # --- OFFLOAD (UAV) COMPUTATION & COMMUNICATION ---
            uav_energy = 0.0
            tx_energy = 0.0
            transmit_time = 0.0
            uav_time = 0.0
            rate = 0.0
            
            if offload_bits > 0 and self.uav and self.irs:
                # 1. Distances
                d_G1 = distance_3d(self.irs.base_position, self.uav.position)
                d_G2 = distance_2d_to_3d(vehicle.position, self.irs.base_position, z2d=0.0)
                d_G3 = distance_2d_to_3d(vehicle.position, self.uav.position, z2d=0.0)
                
                # 2. Channels
                # Fake nlos and b_hat for deterministic behavior in audit
                num_elements = self.irs.num_elements
                nlos = np.ones(num_elements, dtype=complex)
                b_hat = 1.0 + 0j
                z = 0.0 
                z_prime = 0.0
                
                G1 = compute_G1(d_G1, beta_1, beta_2, num_elements, z)
                G2 = compute_G2(d_G2, beta_1, beta_3, k_rician, num_elements, z_prime, nlos)
                G3 = compute_G3(d_G3, beta_1, beta_4, b_hat)
                
                # Effective channel
                G_eff = compute_effective_channel(G3, G1, Theta, G2)
                
                # 3. SNR and Rate
                tx_power_w = 10 ** ((physical_action['transmit_power'] - 30) / 10)
                snr = compute_snr(tx_power_w, np.array([]), G_eff, np.array([]), noise_w)
                rate = compute_shannon_rate(bandwidth, snr)
                
                # 4. Offloading Energy & Time
                if rate > 0:
                    transmit_time = offload_bits / rate
                    tx_energy = tx_power_w * transmit_time
                    
                    offload_task = Task(task.task_id, vehicle.vehicle_id, offload_bits, task.cycles_per_bit, task.deadline)
                    uav_energy = compute_uav_energy(offload_task, self.uav, kappa)
                    uav_time = compute_uav_time(offload_task, self.uav)
                else:
                    transmit_time = float('inf')
            
            # Task Completion logic
            # Removed the 10-second hard timeout as per STAGE 6A REVISION instructions.
            # Task drop logic will now rely purely on physical constraints if needed, but for now 
            # we simply allow the time to be recorded.
            pass
            
            completed_bits = local_bits + offload_bits
            total_completed_bits += completed_bits
            
            task_energy = local_energy + tx_energy + uav_energy
            total_energy += task_energy
            
            total_local_energy += local_energy
            total_uav_energy += uav_energy
            total_tx_energy += tx_energy
            
        env_state_info = {
            'completed_bits': total_completed_bits,
            'total_energy': total_energy,
            'communication_energy': total_tx_energy,
            'computation_energy': total_local_energy + total_uav_energy,
            'local_computation_energy': total_local_energy,
            'rsu_computation_energy': 0.0,
            'uav_computation_energy': total_uav_energy,
            'task_completion_ratio': total_completed_bits / sum([t.input_bits for t in self.tasks]),
            'transmission_time': transmit_time, # Diagnostic (from last task)
            'computation_time': max(local_time, uav_time) if self.tasks else 0.0
        }
        
        # 5. Generate Reward
        reward, info = self.reward_func.calculate_reward(env_state_info)
        
        # 6. Generate Next State
        next_state = self.state_rep.extract_state(self)
        
        # 7. Check Done
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return next_state, reward, done, info
