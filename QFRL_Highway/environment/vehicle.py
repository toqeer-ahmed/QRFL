from dataclasses import dataclass
from typing import Tuple

@dataclass
class Vehicle:
    """
    Represents a Vehicular Terminal (VT) from the published paper.
    """
    vehicle_id: int
    x: float
    y: float
    speed: float  # km/h
    cpu_frequency: float  # GHz
    max_transmit_power: float  # dBm

    @property
    def position(self) -> Tuple[float, float]:
        """Returns the 2D coordinate X_v = [l_v, m_v]"""
        return self.x, self.y
