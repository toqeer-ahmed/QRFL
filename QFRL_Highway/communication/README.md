# Communication Module

This module implements the communication layer for the QFRL reproduction project. 

## Implemented Equations
The following equations from the paper were exactly implemented:

1. **IRS-UAV Channel ($G_1$)**:
   - $G_1 = b_1 \sqrt{\beta_1 x_1^{-\beta_2}}$
   - $b_1 = (1, e^{-jz}, \dots, e^{-jz(P-1)})^T$
   - Implemented in `channel.compute_G1`

2. **VT-IRS Channel ($G_2^v$)**:
   - $G_2^v = b_2^v \sqrt{\beta_1 (x_2^v)^{-\beta_3}}$
   - $b_2^v = \sqrt{\frac{K}{K+1}} G_{2,\text{LoS}}^v + \sqrt{\frac{1}{K+1}} G_{2,\text{NLoS}}^v$
   - $G_{2,\text{LoS}}^v = (1, e^{-jz'}, \dots, e^{-jz'(P-1)})$
   - Implemented in `channel.compute_G2`

3. **UAV-VT Channel ($G_3$)**:
   - $G_3 = \hat{b} \sqrt{\beta_1 (x_3^v)^{-\beta_4}}$
   - Implemented in `channel.compute_G3`

4. **Phase Shift terms**:
   - $z = \frac{2\pi h}{\lambda} \cos(\beta)$
   - $z' = \frac{2\pi h}{\lambda} \cos(\rho)$
   - Implemented in `channel.compute_z`

5. **IRS Reflection Matrix**:
   - $\Theta_\tau = \text{diag}(\alpha_1^\tau, \alpha_2^\tau, \dots, \alpha_P^\tau)$ with $\theta_p = e^{j\omega_p}$
   - Implemented in `channel.compute_reflection_matrix`

6. **Effective Channel**:
   - $G_v = G_3 + G_1 \Theta_\tau G_2^v$
   - *Note on dimensions*: $b_1$ is defined as a column vector ($P \times 1$) and $G_{2,\text{LoS}}^v$ as a row vector ($1 \times P$). $\Theta_\tau$ is $P \times P$. To produce a valid scalar channel gain in code, the bilinear form $G_2 \Theta_\tau G_1$ is calculated explicitly to handle the dimensions.
   - Implemented in `channel.compute_effective_channel`

7. **Signal-to-Noise Ratio (SNR)**:
   - $\text{SNR}_v(k) = \frac{U_v(k) ||G_v||^2}{\sum_{j \neq v} U_j(k) ||G_{vj}||^2 + \psi^2}$
   - Implemented in `snr.compute_snr`

8. **Shannon Transmission Rate**:
   - $R_1^{\max}(k) = B_w \log_2(1 + \text{SNR})$
   - Implemented in `rate.compute_shannon_rate`

## Parameters
### Provided by the Paper
- **VT maximum transmit power**: 20 dBm
- **Noise power** ($\psi^2$): -120 dBm

### Required but Unspecified (Flagged with TODO)
- $\beta_1$ (Reference path loss)
- $\beta_2, \beta_3, \beta_4$ (Path loss exponents)
- $K$ (Rician factor)
- $\lambda$ (Carrier wavelength)
- $h$ (IRS element spacing)
- $B_w$ (Bandwidth)
- $P$ (Number of IRS elements)

### Test-Only Assumptions
- Fixed angles $\beta, \rho$ were used directly in the test rather than calculated via 3D geometry since exact orientation wasn't yet defined.
- $P=4$ was used for testing.
- Generated standard normal complex Gaussian noise for NLoS fading factors in tests.
