from dataclasses import dataclass
from typing import Optional

@dataclass
class Task:
    """
    Represents a computational task originating from a vehicle, based on the published paper.
    """
    task_id: int
    originating_vehicle_id: int
    input_bits: float  # Input data size in bits
    cycles_per_bit: float  # Required CPU cycles per input bit
    deadline: Optional[float] = None  # Deadline in seconds
    
    def __post_init__(self):
        if self.input_bits < 0:
            raise ValueError("Task input_bits cannot be negative.")
        if self.cycles_per_bit < 0:
            raise ValueError("Task cycles_per_bit cannot be negative.")
            
    @property
    def total_cycles(self) -> float:
        """Total required CPU cycles (C) = input_bits * cycles_per_bit"""
        return self.input_bits * self.cycles_per_bit
