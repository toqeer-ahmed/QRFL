from environment.task import Task
from environment.rsu import RSU
from environment.uav import UAV

def compute_rsu_time(task: Task, rsu: RSU) -> float:
    """
    Computes RSU/edge computation time.
    T_RSU = C / f_r
    f_r is in GHz, so we convert it to Hz internally.
    """
    if rsu.cpu_frequency <= 0:
        raise ValueError("RSU CPU frequency must be strictly positive.")
        
    f_r_hz = rsu.cpu_frequency * 1e9
    return task.total_cycles / f_r_hz

def compute_uav_time(task: Task, uav: UAV) -> float:
    """
    Computes UAV computation time.
    T_UAV = C / f_u
    f_u is in GHz, so we convert it to Hz internally.
    """
    if uav.cpu_frequency <= 0:
        raise ValueError("UAV CPU frequency must be strictly positive.")
        
    f_u_hz = uav.cpu_frequency * 1e9
    return task.total_cycles / f_u_hz
