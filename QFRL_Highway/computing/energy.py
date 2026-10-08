from environment.task import Task
from environment.vehicle import Vehicle
from environment.rsu import RSU
from environment.uav import UAV

def compute_energy(task: Task, cpu_frequency_ghz: float, kappa: float) -> float:
    """
    Computes energy consumed by local/edge computation.
    E = kappa * C * f^2
    cpu_frequency_ghz is in GHz, so we convert it to Hz internally.
    """
    if cpu_frequency_ghz <= 0:
        raise ValueError("CPU frequency must be strictly positive.")
        
    f_hz = cpu_frequency_ghz * 1e9
    return kappa * task.total_cycles * (f_hz ** 2)

def compute_local_energy(task: Task, vehicle: Vehicle, kappa: float) -> float:
    """Computes local computing energy for a VT"""
    return compute_energy(task, vehicle.cpu_frequency, kappa)

def compute_rsu_energy(task: Task, rsu: RSU, kappa: float) -> float:
    """Computes computation energy for an RSU"""
    return compute_energy(task, rsu.cpu_frequency, kappa)

def compute_uav_energy(task: Task, uav: UAV, kappa: float) -> float:
    """Computes computation energy for the UAV"""
    return compute_energy(task, uav.cpu_frequency, kappa)
