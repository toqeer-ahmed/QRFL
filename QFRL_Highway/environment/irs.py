from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class IRSElement:
    """
    Represents a single element of the IRS with a 3D coordinate X_p = [l_p, m_p, n_p].
    """
    element_id: int
    x: float
    y: float
    z: float
    phase: float = 0.0

@dataclass
class IRS:
    """
    Represents the Intelligent Reflecting Surface (IRS) from the published paper.
    Contains P passive components.
    """
    num_elements: int
    base_x: float
    base_y: float
    base_z: float
    elements: List[IRSElement]
    
    @property
    def base_position(self) -> Tuple[float, float, float]:
        return self.base_x, self.base_y, self.base_z
