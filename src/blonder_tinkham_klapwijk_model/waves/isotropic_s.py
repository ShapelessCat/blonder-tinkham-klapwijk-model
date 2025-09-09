from typing import Final

from blonder_tinkham_klapwijk_model.config.gap_config import IsotropicGapConfig
from blonder_tinkham_klapwijk_model.transparency import (
    normal_transparency_of,
    superconductor_transparency_of,
)
from . import gamma_function_of
from blonder_tinkham_klapwijk_model.config.formula_type import S0Formula, S1Formula


def isotropic_s(
    energy: float,
    broadening_parameter: float,
    barrier_strength: float,
    gap_config: IsotropicGapConfig,
    normalization_conductance_factor: float,
    formula_type: S0Formula | S1Formula,
) -> float:
    """Calculates the integral function for superconducting transport calculations.

    Parameters
    ----------
    energy : float
        Energy value(s) in eV (can be scalar or array)

    broadening_parameter : float
        Broadening parameter in eV

    barrier_strength : float
        Barrier strength (dimensionless)

    gap_config : AnisotropicGapConfig
       Superconducting (gap_plus, gap_minus) in eV

    normalization_conductance_factor : float
        Normalized conductivity reference value

    Returns
    -------
        Calculated conductivity function value(s)
    """
    delta_plus = gap_config.gap
    delta_minus = gap_config.gap

    # Complex energy with broadening
    complex_energy: Final[complex] = energy + 1j * broadening_parameter
    gamma_plus = gamma_function_of(complex_energy, delta_plus)
    gamma_minus = gamma_function_of(complex_energy, delta_minus)

    normal_transparency = normal_transparency_of(1.0, barrier_strength)
    superconductor_transparency = superconductor_transparency_of(
        normal_transparency, gamma_plus, gamma_minus
    )
    return (
        superconductor_transparency * normal_transparency / normalization_conductance_factor
    ).real
