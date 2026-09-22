from dataclasses import dataclass
from typing import Tuple

@dataclass
class UAV:
    """
    Represents the Unmanned Aerial Vehicle (UAV) from the published paper.
    Altitude is fixed during this simulation stage.
    """
    x: float
    y: float
    altitude: float  # meters
    max_speed: float  # km/h
    cpu_frequency: float  # GHz

    @property
    def position(self) -> Tuple[float, float, float]:
        """Returns the 3D coordinate X_u = (l_u, m_u, H)"""
        return self.x, self.y, self.altitude
