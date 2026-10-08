import numpy as np
import yaml
import sys
import os

sys.path.append(os.path.dirname(__file__))

from environment.vehicle import Vehicle
from environment.uav import UAV
from environment.rsu import RSU
from environment.task import Task
from computing.local import compute_local_time
from computing.edge import compute_rsu_time, compute_uav_time
from computing.energy import compute_local_energy, compute_rsu_energy, compute_uav_energy

def test():
    # 1. Load config
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'paper_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract CPU params
    f_v = config['vehicles']['cpu_frequency_ghz']  # 4 GHz
    f_r = config['rsu']['cpu_frequency_ghz']       # 9 GHz
    f_u = config['uav']['cpu_frequency_ghz']       # 12 GHz
    
    kappa = config['computation']['kappa']
    input_bits = config['tasks']['input_bits']
    cycles_per_bit = config['tasks']['cycles_per_bit']

    vt = Vehicle(vehicle_id=1, x=0.0, y=0.0, speed=0.0, cpu_frequency=f_v, max_transmit_power=20.0)
    rsu = RSU(rsu_id=1, x=0.0, y=0.0, cpu_frequency=f_r)
    uav = UAV(x=0.0, y=0.0, altitude=300.0, max_speed=0.0, cpu_frequency=f_u)
    
    task = Task(task_id=1, originating_vehicle_id=1, input_bits=input_bits, cycles_per_bit=cycles_per_bit)
    
    print("=== STAGE 4 COMPUTING TEST ===")
    
    t_local = compute_local_time(task, vt)
    t_rsu = compute_rsu_time(task, rsu)
    t_uav = compute_uav_time(task, uav)
    
    print(f"Task total cycles: {task.total_cycles:.2e}")
    print(f"Local time (4 GHz): {t_local:.4e} s")
    print(f"RSU time (9 GHz): {t_rsu:.4e} s")
    print(f"UAV time (12 GHz): {t_uav:.4e} s")
    
    # D. Basic Sanity Check
    assert t_local > t_rsu > t_uav, "Faster processors should take less time"
    
    # E. Scaling Check
    task_double = Task(task_id=2, originating_vehicle_id=1, input_bits=input_bits * 2, cycles_per_bit=cycles_per_bit)
    t_local_double = compute_local_time(task_double, vt)
    assert np.isclose(t_local_double, 2 * t_local), "Doubling task size should double computation time"
    
    # F. Frequency Check
    vt_fast = Vehicle(vehicle_id=2, x=0.0, y=0.0, speed=0.0, cpu_frequency=f_v * 2, max_transmit_power=20.0)
    t_local_fast = compute_local_time(task, vt_fast)
    assert np.isclose(t_local_fast, t_local / 2), "Doubling frequency should halve computation time"
    
    # G. Energy Checks
    e_local = compute_local_energy(task, vt, kappa)
    e_rsu = compute_rsu_energy(task, rsu, kappa)
    e_uav = compute_uav_energy(task, uav, kappa)
    
    print(f"\nLocal energy: {e_local:.4e} J")
    print(f"RSU energy: {e_rsu:.4e} J")
    print(f"UAV energy: {e_uav:.4e} J")
    
    assert e_local > 0 and np.isfinite(e_local), "Local energy must be finite and non-negative"
    assert e_rsu > 0 and np.isfinite(e_rsu), "RSU energy must be finite and non-negative"
    assert e_uav > 0 and np.isfinite(e_uav), "UAV energy must be finite and non-negative"
    
    # Validation for negative inputs
    try:
        Task(task_id=3, originating_vehicle_id=1, input_bits=-100, cycles_per_bit=100)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for negative input bits"
        
    try:
        vt_invalid = Vehicle(vehicle_id=3, x=0.0, y=0.0, speed=0.0, cpu_frequency=-1.0, max_transmit_power=20.0)
        compute_local_time(task, vt_invalid)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for negative CPU frequency"
    
    print("\nAll validation checks passed successfully!")

if __name__ == "__main__":
    test()
