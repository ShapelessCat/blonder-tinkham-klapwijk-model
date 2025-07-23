from typing import Final

import numpy as np
from numpy import sqrt, conj, exp, complex128


def isotropic_wave(
        energy: float,
        gamma: float,
        barrier_strength: float,
        gap: float,
) -> float:
    """Calculate the integrand function for s-wave superconductivity (BTK theory)

    Parameters:
    energy : float
        Energy value(s) in eV (can be scalar or array)

    gamma : float
        Broadening parameter in eV

    barrier_strength : float
        Barrier strength (dimensionless)

    gap : float
        Superconducting gap (mV) ???
        Superconducting gap in eV ???

    Returns:
    float
        The calculated integrand value(s)
    """
    # Complex energy with broadening: complex128(energy) + 1j * gamma
    complex_energy: Final[complex] = energy + 1j * gamma

    # Calculate u^2 and v^2 (coherence factors)
    with np.errstate(divide='ignore', invalid='ignore'):
        u2 = 0.5 * (1 + sqrt((complex_energy**2 - gap**2) / complex_energy**2))
    v2 = 1 - u2

    # Denominator and normalization factor
    denom = u2 + (u2 - v2) * barrier_strength ** 2
    sigmanorm = 1 / (1 + barrier_strength ** 2)

    # Calculate a and b coefficients
    a = sqrt(u2) * sqrt(v2) / denom
    b = -(u2 - v2) * complex128(barrier_strength ** 2, barrier_strength) / denom

    # Calculate probabilities
    aa = conj(a) * a
    bb = conj(b) * b

    # Final integrand calculation
    return (1 + aa - bb) / sigmanorm
    # TODO: IMPORTANT
    # # Handle any NaN/inf values that might occur
    # y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # return np.real(y)  # Return real part for physical results


def integr_func_s_wave(energy, vv, temperature, gamma, barrier_strength, gap):
    """
    Calculate the integrand function for s-wave superconductivity (BTK theory)

    Parameters:
    e : float or ndarray
        Energy value(s) (mV)
    vv : float
        Bias voltage (mV)
    temperature : float
        Temperature (K)
    gamma : float
        Broadening parameter
    barrier : float
        Barrier strength parameter (Z)
    gap : float
        Superconducting gap (mV)

    Returns:
    y : float or ndarray
        The calculated integrand value(s)
    """
    # Physical constants
    kb = 1.38066e-23       # Boltzmann constant (J/K)
    elec = 1.60228e-19     # Electron charge (C)
    c = elec / (kb * 1000)  # Unit conversion factor

    # Complex energy with broadening
    e_complex = complex128(energy) + 1j * gamma

    # Calculate u^2 and v^2 (coherence factors)
    with np.errstate(divide='ignore', invalid='ignore'):
        u2 = 0.5 * (1 + sqrt((e_complex**2 - gap**2) / e_complex**2))
    v2 = 1 - u2

    # Denominator and normalization factor
    denom = u2 + (u2 - v2) * barrier_strength ** 2
    sigmanorm = 1 / (1 + barrier_strength ** 2)

    # Calculate a and b coefficients
    a = sqrt(u2) * sqrt(v2) / denom
    b = -(u2 - v2) * complex128(barrier_strength ** 2, barrier_strength) / denom

    # Calculate probabilities
    aa = conj(a) * a
    bb = conj(b) * b

    # Fermi functions
    exp0 = exp(c * energy / temperature)
    expv = exp(c * (energy - vv) / temperature)
    fermi0 = 1 / (1 + exp0)
    fermiv = 1 / (1 + expv)

    # Final integrand calculation
    y = (fermiv - fermi0) * (1 + aa - bb) / sigmanorm

    # Handle any NaN/inf values that might occur
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return np.real(y)  # Return real part for physical results
