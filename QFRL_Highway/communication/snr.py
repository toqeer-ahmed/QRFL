import numpy as np

def compute_snr(target_tx_power_w: float, 
                interfering_tx_powers_w: np.ndarray, 
                target_eff_channel: complex, 
                interfering_eff_channels: np.ndarray, 
                noise_power_w: float) -> float:
    """
    Calculates SNR.
    Paper Eq: SNR_v(k) = [ U_v(k) ||G_v||^2 ] / [ sum_{j != v} U_j(k) ||G_vj||^2 + psi^2 ]
    
    target_tx_power_w: Transmit power of target VT in Watts
    interfering_tx_powers_w: Array of transmit powers of interfering VTs in Watts
    target_eff_channel: Effective channel G_v for the target VT (scalar complex)
    interfering_eff_channels: Array of effective channels G_vj for interfering VTs
    noise_power_w: Noise power in Watts (psi^2)
    """
    signal_power = target_tx_power_w * (np.abs(target_eff_channel) ** 2)
    
    interference_power = 0.0
    for p_int, g_int in zip(interfering_tx_powers_w, interfering_eff_channels):
        interference_power += p_int * (np.abs(g_int) ** 2)
        
    snr = signal_power / (interference_power + noise_power_w)
    return float(snr)
