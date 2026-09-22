import numpy as np
from typing import Tuple

def distance_2d(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculates 2D Euclidean distance."""
    return float(np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2))

def distance_3d(pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
    """Calculates 3D Euclidean distance."""
    return float(np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2))

def distance_2d_to_3d(pos_2d: Tuple[float, float], pos_3d: Tuple[float, float, float], z2d: float = 0.0) -> float:
    """
    Calculates distance between a 2D point (assumed to be at z=z2d) and a 3D point.
    Useful for VT (ground) to UAV/IRS distances.
    """
    return float(np.sqrt((pos_2d[0] - pos_3d[0])**2 + (pos_2d[1] - pos_3d[1])**2 + (z2d - pos_3d[2])**2))
