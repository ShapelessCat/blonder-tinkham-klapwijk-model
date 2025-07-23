from typing import Final

import numpy as np
from numpy import sqrt, conj, exp, complex128


def isotropic_wave(
        energy: float,
        broadening_parameter: float,
        barrier_strength: float,
        gap: float,
        normalization_conductance_factor: float,
) -> float:
    """Calculate the integrand function for s-wave superconductivity (BTK theory)

    Parameters:
    energy : float
        Energy value(s) in eV (can be scalar or array)

    broadening_parameter : float
        Broadening parameter in eV

    barrier_strength : float
        Barrier strength (dimensionless)

    gap : float
        Superconducting gap (mV) ???
        Superconducting gap in eV ???

    normalization_conductance_factor: float
        Normalized conductivity reference value

    Returns:
    float
        The calculated integrand value(s)
    """
    # Complex energy with broadening:
    complex_energy: Final[complex] = complex128(energy) + 1j * broadening_parameter

    # Calculate u^2 and v^2 (coherence factors)
    with np.errstate(divide='ignore', invalid='ignore'):
        u2 = 0.5 * (1 + sqrt((complex_energy**2 - gap**2) / complex_energy**2))
    v2 = 1 - u2

    # Denominator and normalization factor
    denom = u2 + (u2 - v2) * barrier_strength ** 2
    sigmanorm = normalization_conductance_factor

    # Calculate a and b coefficients
    a = sqrt(u2) * sqrt(v2) / denom
    b = -(u2 - v2) * complex128(barrier_strength ** 2, barrier_strength) / denom

    # Calculate probabilities
    aa = conj(a) * a
    bb = conj(b) * b

    # Final integrand calculation
    result = np.nan_to_num((1 + aa - bb) / sigmanorm, nan=0.0, posinf=0.0, neginf=0.0)
    return np.real(result)
