import numpy as np

def dbm_to_watts(dbm: float) -> float:
    """Convert power in dBm to Watts."""
    return 10 ** ((dbm - 30) / 10)

def watts_to_dbm(watts: float) -> float:
    """Convert power in Watts to dBm."""
    return 10 * np.log10(watts + 1e-15) + 30

def compute_z(h: float, wavelength: float, angle_rad: float) -> float:
    """Computes phase progression z or z' given spacing, wavelength, and angle."""
    return (2 * np.pi * h / wavelength) * np.cos(angle_rad)

def compute_G1(distance: float, beta_1: float, beta_2: float, num_elements: int, z: float) -> np.ndarray:
    """
    IRS-UAV channel.
    Paper Eq: G_1 = b_1 * sqrt(beta_1 * x_1^(-beta_2))
    b_1 = (1, e^(-jz), ..., e^(-jz(P-1)))^T
    Returns column vector of shape (P, 1).
    """
    indices = np.arange(num_elements)
    b_1 = np.exp(-1j * z * indices)
    path_loss_amp = np.sqrt(beta_1 * (distance ** (-beta_2)))
    G_1 = b_1 * path_loss_amp
    return G_1.reshape(num_elements, 1)

def compute_G2(distance: float, beta_1: float, beta_3: float, K: float, num_elements: int, z_prime: float, nlos_fading: np.ndarray) -> np.ndarray:
    """
    VT-IRS channel.
    Paper Eq: G_2^v = b_2^v * sqrt(beta_1 * (x_2^v)^(-beta_3))
    b_2^v = sqrt(K/(K+1)) * G_2,LoS^v + sqrt(1/(K+1)) * G_2,NLoS^v
    G_2,LoS = (1, e^(-jz'), ..., e^(-jz'(P-1)))
    Returns row vector of shape (1, P).
    """
    indices = np.arange(num_elements)
    G_2_LoS = np.exp(-1j * z_prime * indices)
    
    b_2_v = np.sqrt(K / (K + 1)) * G_2_LoS + np.sqrt(1 / (K + 1)) * nlos_fading
    path_loss_amp = np.sqrt(beta_1 * (distance ** (-beta_3)))
    
    G_2 = b_2_v * path_loss_amp
    return G_2.reshape(1, num_elements)

def compute_G3(distance: float, beta_1: float, beta_4: float, b_hat: complex) -> complex:
    """
    UAV-VT channel.
    Paper Eq: G_3 = b_hat * sqrt(beta_1 * (x_3^v)^(-beta_4))
    """
    path_loss_amp = np.sqrt(beta_1 * (distance ** (-beta_4)))
    return b_hat * path_loss_amp

def compute_reflection_matrix(phases: np.ndarray) -> np.ndarray:
    """
    IRS reflection-coefficient matrix.
    Paper Eq: Theta_tau = diag(alpha_1^tau, alpha_2^tau, ..., alpha_P^tau)
    where alpha_p = exp(j * omega_p).
    """
    return np.diag(np.exp(1j * phases))

def compute_effective_channel(G3: complex, G1: np.ndarray, Theta: np.ndarray, G2: np.ndarray) -> complex:
    """
    Effective channel.
    Paper Eq: G_v = G_3 + G_1 Theta_tau G_2^v
    NOTE: Handled carefully for valid matrix dimensions to produce a scalar (1x1) channel gain.
    Given b_1 is a column vector and G_2,LoS is a row vector, G2 @ Theta @ G1 yields the scalar sum.
    """
    assert Theta.shape[0] == Theta.shape[1], "Theta must be a square matrix"
    P = Theta.shape[0]
    G1_vec = G1.flatten()
    G2_vec = G2.flatten()
    
    assert len(G1_vec) == P, f"G1 must have P elements, got {len(G1_vec)}"
    assert len(G2_vec) == P, f"G2 must have P elements, got {len(G2_vec)}"
    
    irs_path = G2_vec @ Theta @ G1_vec
    return complex(G3 + irs_path)
