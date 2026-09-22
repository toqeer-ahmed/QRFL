from dataclasses import dataclass
from typing import Tuple

@dataclass
class RSU:
    """
    Represents a Road Side Unit (RSU) from the published paper.
    """
    rsu_id: int
    x: float
    y: float
    cpu_frequency: float  # GHz

    @property
    def position(self) -> Tuple[float, float]:
        """Returns the 2D coordinate X_r = [l_r, m_r]"""
        return self.x, self.y
