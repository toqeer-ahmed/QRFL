import numpy as np
import yaml
import sys
import os

sys.path.append(os.path.dirname(__file__))

from environment.vehicle import Vehicle
from environment.uav import UAV
from environment.irs import IRS, IRSElement
from communication.distance import distance_2d, distance_3d, distance_2d_to_3d
from communication.channel import (
    dbm_to_watts, watts_to_dbm,
    compute_z, compute_G1, compute_G2, compute_G3,
    compute_reflection_matrix, compute_effective_channel
)
from communication.snr import compute_snr
from communication.rate import compute_shannon_rate

def test():
    np.random.seed(42) # For deterministic repeatability
    
    # 1. Load config
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'paper_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    comm_cfg = config['communication']
    
    # Paper specified params
    noise_dbm = comm_cfg['noise_power_dbm']
    noise_w = dbm_to_watts(noise_dbm)
    
    # Test-only / Unspecified params
    beta_1 = comm_cfg['beta_1']
    beta_2 = comm_cfg['beta_2']
    beta_3 = comm_cfg['beta_3']
    beta_4 = comm_cfg['beta_4']
    K = comm_cfg['K_rician']
    wavelength = comm_cfg['wavelength_m']
    h_spacing = comm_cfg['irs_element_spacing_m']
    bandwidth = comm_cfg['bandwidth_hz']
    
    # 2. Setup objects
    vt1 = Vehicle(vehicle_id=1, x=10.0, y=10.0, speed=0.0, cpu_frequency=4.0, max_transmit_power=20.0)
    vt2 = Vehicle(vehicle_id=2, x=50.0, y=50.0, speed=0.0, cpu_frequency=4.0, max_transmit_power=20.0) # Interferer
    uav = UAV(x=0.0, y=0.0, altitude=300.0, max_speed=0.0, cpu_frequency=12.0)
    
    num_elements = 4 # Small number for testing (unspecified)
    elements = [IRSElement(i, 0.0, 0.0, 0.0) for i in range(num_elements)]
    irs = IRS(num_elements=num_elements, base_x=20.0, base_y=20.0, base_z=50.0, elements=elements)
    
    print("=== STAGE 3 COMMUNICATION TEST ===")
    
    # 3. Distances
    x1 = distance_3d(irs.base_position, uav.position) # IRS-UAV
    x2_v1 = distance_2d_to_3d(vt1.position, irs.base_position) # VT1-IRS
    x3_v1 = distance_2d_to_3d(vt1.position, uav.position) # UAV-VT1
    
    x2_v2 = distance_2d_to_3d(vt2.position, irs.base_position) # VT2-IRS
    x3_v2 = distance_2d_to_3d(vt2.position, uav.position) # UAV-VT2
    
    print(f"\nDistances:")
    print(f" IRS-UAV (x1): {x1:.2f} m")
    print(f" VT1-IRS (x2_v1): {x2_v1:.2f} m")
    print(f" UAV-VT1 (x3_v1): {x3_v1:.2f} m")
    
    # 4. Angles (using dummy angles for test as mobility/geometry isn't fully defined)
    angle_beta = np.pi / 4  
    angle_rho1 = np.pi / 3  
    angle_rho2 = np.pi / 6  
    
    z = compute_z(h_spacing, wavelength, angle_beta)
    z_prime1 = compute_z(h_spacing, wavelength, angle_rho1)
    z_prime2 = compute_z(h_spacing, wavelength, angle_rho2)
    
    # 5. Channel Gains
    G1 = compute_G1(x1, beta_1, beta_2, num_elements, z)
    
    # NLoS fading (complex gaussian)
    nlos1 = (np.random.randn(num_elements) + 1j * np.random.randn(num_elements)) / np.sqrt(2)
    nlos2 = (np.random.randn(num_elements) + 1j * np.random.randn(num_elements)) / np.sqrt(2)
    
    G2_v1 = compute_G2(x2_v1, beta_1, beta_3, K, num_elements, z_prime1, nlos1)
    G2_v2 = compute_G2(x2_v2, beta_1, beta_3, K, num_elements, z_prime2, nlos2)
    
    b_hat1 = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    b_hat2 = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    
    G3_v1 = compute_G3(x3_v1, beta_1, beta_4, b_hat1)
    G3_v2 = compute_G3(x3_v2, beta_1, beta_4, b_hat2)
    
    print(f"\nChannel Gains (|G|):")
    print(f" |G3| (UAV-VT1 direct): {np.abs(G3_v1):.2e}")
    print(f" |G1| (IRS-UAV): {np.abs(G1).mean():.2e} (mean over elements)")
    print(f" |G2_v1| (VT1-IRS): {np.abs(G2_v1).mean():.2e} (mean over elements)")
    
    # 6. IRS reflection
    phases = np.random.uniform(0, 2*np.pi, num_elements)
    Theta = compute_reflection_matrix(phases)
    print(f"\nIRS Reflection Matrix (diag elements magnitude): {np.abs(np.diag(Theta))}")
    
    # 7. Effective Channels
    G_eff_v1 = compute_effective_channel(G3_v1, G1, Theta, G2_v1)
    G_eff_v2 = compute_effective_channel(G3_v2, G1, Theta, G2_v2)
    
    print(f"\nEffective Channels:")
    print(f" VT1 |G_eff|: {np.abs(G_eff_v1):.2e}")
    print(f" VT2 |G_eff|: {np.abs(G_eff_v2):.2e}")
    
    # Verify effective channel dimensions / value logically
    # Changing phases should change effective channel
    phases_alt = np.random.uniform(0, 2*np.pi, num_elements)
    Theta_alt = compute_reflection_matrix(phases_alt)
    G_eff_v1_alt = compute_effective_channel(G3_v1, G1, Theta_alt, G2_v1)
    assert not np.isclose(np.abs(G_eff_v1), np.abs(G_eff_v1_alt)), "Changing phases did not change effective channel"
    
    # Verify distance effect on channel gains
    G1_far = compute_G1(x1 * 2, beta_1, beta_2, num_elements, z)
    assert np.abs(G1_far).mean() < np.abs(G1).mean(), "Channel gain did not decrease with distance"
    
    # 8. SNR
    p_tx_w = dbm_to_watts(vt1.max_transmit_power)
    interferers_w = np.array([dbm_to_watts(vt2.max_transmit_power)])
    interferers_g = np.array([G_eff_v2])
    
    snr_linear = compute_snr(p_tx_w, interferers_w, G_eff_v1, interferers_g, noise_w)
    print(f"\nSNR: {snr_linear:.2e} (linear), {10*np.log10(snr_linear):.2f} dB")
    
    # Verify noise effect
    snr_noisy = compute_snr(p_tx_w, interferers_w, G_eff_v1, interferers_g, noise_w * 10)
    assert snr_noisy < snr_linear, "Increasing noise did not decrease SNR"
    
    # Verify TX power effect (no interference change)
    snr_high_tx = compute_snr(p_tx_w * 2, interferers_w, G_eff_v1, interferers_g, noise_w)
    assert snr_high_tx > snr_linear, "Increasing TX power did not increase SNR"
    
    # 9. Rate
    rate = compute_shannon_rate(bandwidth, snr_linear)
    print(f"Transmission Rate: {rate/1e6:.2f} Mbps")
    
    # Verify rate vs SNR
    rate_high = compute_shannon_rate(bandwidth, snr_high_tx)
    assert rate_high > rate, "Increasing SNR did not increase rate"

if __name__ == "__main__":
    test()
