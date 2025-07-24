from typing import Final

import numpy as np

from .transparency import normal_transparency_of, superconductor_transparency_of


def anisotropic_wave(
    theta: float,
    energy: float,
    broadening_parameter: float,
    barrier_strength: float,
    gap: float,
    angle: float,
    normalization_conductance_factor: float,
) -> float:
    """Calculates the integral function for superconducting transport calculations.

    Parameters
    ----------
    theta: float
        Angle(s) in radians (can be scalar or array)

    energy: float
        Energy value(s) in eV (can be scalar or array)

    broadening_parameter: float
        Broadening parameter in eV

    barrier_strength: float
        Barrier strength (dimensionless)

    gap: float
       Superconducting gap in eV

    angle: float
        Crystallographic orientation angle in degrees

    normalization_conductance_factor: float
        Normalized conductivity reference value

    Returns
    -------
        Calculated conductivity function value(s)
    """
    delta_plus = gap * np.cos(2 * (theta - np.deg2rad(angle))) ** 4
    delta_minus = gap * np.cos(2 * (-theta - np.deg2rad(angle))) ** 4

    # Complex energy with broadening
    complex_energy: Final[complex] = energy + 1j * broadening_parameter
    gamma_plus = gamma_function_of(complex_energy, delta_plus)
    gamma_minus = gamma_function_of(complex_energy, delta_minus)

    cos_theta = np.cos(theta)
    normal_transparency = normal_transparency_of(cos_theta, barrier_strength)
    superconductor_transparency = superconductor_transparency_of(
        normal_transparency, gamma_plus, gamma_minus
    )
    return (
        superconductor_transparency
        * normal_transparency
        * cos_theta
        / normalization_conductance_factor
    )


def gamma_function_of(energy: complex, delta: float) -> complex:
    return energy / np.abs(delta) - np.sqrt((energy / np.abs(delta)) ** 2 - 1)
