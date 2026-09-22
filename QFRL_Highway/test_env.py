import yaml
import sys
import os

# Ensure the environment modules can be imported
sys.path.append(os.path.dirname(__file__))

from environment.environment import QFRLEnvironment

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'paper_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    env = QFRLEnvironment()
    env.initialize_from_config(config)
    
    print("=== Environment Initialization Test ===")
    
    print(f"\n1. Vehicles (Count: {len(env.vehicles)})")
    if env.vehicles:
        v = env.vehicles[0]
        print(f"Sample Vehicle -> ID: {v.vehicle_id}, Position: {v.position}, " 
              f"Max Speed: {v.speed} km/h, CPU: {v.cpu_frequency} GHz, "
              f"Max TX Power: {v.max_transmit_power} dBm")
              
    print(f"\n2. RSUs (Count: {len(env.rsus)})")
    if env.rsus:
        r = env.rsus[0]
        print(f"Sample RSU -> ID: {r.rsu_id}, Position: {r.position}, "
              f"CPU: {r.cpu_frequency} GHz")
              
    print(f"\n3. UAV")
    if env.uav:
        print(f"UAV -> Position: {env.uav.position} (Altitude: {env.uav.altitude} m), "
              f"Max Speed: {env.uav.max_speed} km/h, CPU: {env.uav.cpu_frequency} GHz")
              
    print(f"\n4. IRS")
    if env.irs:
        print(f"IRS -> Base Position: {env.irs.base_position}, "
              f"Elements Count: {env.irs.num_elements}")

if __name__ == "__main__":
    main()
