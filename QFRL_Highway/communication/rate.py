import numpy as np

def compute_shannon_rate(bandwidth_hz: float, snr: float) -> float:
    """
    Calculates transmission rate using Shannon's formula.
    Paper Eq: R_1^max(k) = B_w log2(1 + SNR)
    """
    return bandwidth_hz * np.log2(1 + snr)
