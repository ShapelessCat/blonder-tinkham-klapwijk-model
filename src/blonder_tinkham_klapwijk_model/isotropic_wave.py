import numpy as np
from typing import Union


# Fundamental constants
_BOLTZMANN_CONSTANT_EV_PER_K = 8.617333e-5  # eV/K
_ENERGY_TO_TEMPERATURE_FACTOR = 1 / (_BOLTZMANN_CONSTANT_EV_PER_K * 1000)


def calculate_btk_conductance(
        energy: Union[float, np.ndarray],
        voltage: float,
        temperature: float,
        quasiparticle_broadening: float,
        barrier_strength: float,
        superconducting_gap: float
) -> Union[float, np.ndarray]:
    """
    Calculates the normalized differential conductance using Blonder-Tinkham-Klapwijk (BTK) theory.

    Args:
        energy: Input energy values (eV), can be scalar or array
        voltage: Applied bias voltage (eV)
        temperature: Measurement temperature (K)
        quasiparticle_broadening: Phenomenological broadening parameter (eV)
        barrier_strength: Dimensionless barrier parameter (Z)
        superconducting_gap: Superconducting energy gap (eV)

    Returns:
        Normalized differential conductance (unitless)
    """
    # Complex energy accounting for quasiparticle lifetime
    complex_energy = energy + 1j * quasiparticle_broadening

    # Bogoliubov coherence factors
    electron_like_coherence = 0.5 * (1 + np.sqrt((complex_energy**2 - superconducting_gap**2) / complex_energy**2))
    hole_like_coherence = 1 - electron_like_coherence

    # Andreev and normal reflection amplitudes
    denominator = electron_like_coherence + (electron_like_coherence - hole_like_coherence) * barrier_strength**2

    andreev_reflection_amplitude = np.sqrt(electron_like_coherence * hole_like_coherence) / denominator
    normal_reflection_amplitude = -(electron_like_coherence - hole_like_coherence) * (barrier_strength**2 + 1j*barrier_strength) / denominator

    # Probability densities
    andreev_probability = np.abs(andreev_reflection_amplitude)**2
    normal_reflection_probability = np.abs(normal_reflection_amplitude)**2

    # Fermi-Dirac distributions
    fermi_zero_bias = 1 / (1 + np.exp(_ENERGY_TO_TEMPERATURE_FACTOR * energy / temperature))
    fermi_at_voltage = 1 / (1 + np.exp(_ENERGY_TO_TEMPERATURE_FACTOR * (energy - voltage) / temperature))

    # Normalization and conductance calculation
    normal_state_conductance = 1 / (1 + barrier_strength**2)
    differential_conductance = (fermi_at_voltage - fermi_zero_bias) * (1 + andreev_probability - normal_reflection_probability) / normal_state_conductance

    return differential_conductance