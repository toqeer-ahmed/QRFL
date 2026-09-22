from dataclasses import dataclass
from typing import Optional

@dataclass
class Task:
    """
    Represents a computational task originating from a vehicle, based on the published paper.
    Specific numerical implementations for offloading are deferred to a later stage.
    """
    task_id: int
    originating_vehicle_id: int
    input_data_size: float  # To be properly specified later (Bits/Bytes)
    required_cpu_cycles: float  # To be properly specified later (Cycles)
    deadline: Optional[float] = None  # To be properly specified later (Seconds)
