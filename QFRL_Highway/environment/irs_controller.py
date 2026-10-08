import numpy as np
from typing import List

class IRSController:
    """
    Abstracts the generation of IRS phases.
    """
    def __init__(self, num_elements: int = 50):
        self.num_elements = num_elements
        
    def generate_phase_vector(self, action_scalar: float) -> List[float]:
        """
        IMPLEMENTATION ASSUMPTION:
        Currently maps the single continuous scalar action to a common phase 
        applied to all 50 IRS elements.
        
        Args:
            action_scalar: A single phase value in [-pi, pi]
            
        Returns:
            List of length num_elements containing the phase for each element.
        """
        return [float(action_scalar) for _ in range(self.num_elements)]
