from environment.task import Task
from environment.vehicle import Vehicle

def compute_local_time(task: Task, vehicle: Vehicle) -> float:
    """
    Computes local computation time for a vehicle.
    T_local = C / f_v
    f_v is in GHz, so we convert it to Hz internally.
    """
    if vehicle.cpu_frequency <= 0:
        raise ValueError("Vehicle CPU frequency must be strictly positive.")
    
    f_v_hz = vehicle.cpu_frequency * 1e9
    return task.total_cycles / f_v_hz
