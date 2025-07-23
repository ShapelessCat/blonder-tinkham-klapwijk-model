from functools import partial
from typing import Final

import numpy as np

_BOLTZMANN_CONSTANT: Final[float] = 0.08617  # in meV/K
_MILLIVOLT_TO_meV: Final[float] = 1.0        # 1 mV ≈ 1 meV in natural units

def fermi_window_for_tunneling(
        energy: float,        # in meV/K
        bias_voltage: float,  # in mV
        temperature: float    # in K
) -> float:
    """Computes the Fermi window f(E-eV) - f(E) for tunneling spectroscopy.

    Parameters
    ----------
    energy : float
        Energy values relative to Fermi level (meV)
    bias_voltage : float
        Experimentally applied bias voltage (mV)
    temperature : float
        Measurement temperature (K)

    Returns
    -------
    Fermi window value
    """
    # Convert bias to energy units (1 mV ≈ 1 meV)
    bias_energy = bias_voltage * _MILLIVOLT_TO_meV

    # Fixed temperature Fermi-Dirac distribution function
    partial_fdd = partial(_fermi_dirac_distribution, temperature=temperature)

    shifted_fermi = partial_fdd(energy - bias_energy)
    unshifted_fermi = partial_fdd(energy)
    return shifted_fermi - unshifted_fermi


def _fermi_dirac_distribution(
        energy: float,
        temperature: float
) -> float:
    """Computes the Fermi-Dirac distribution f(E) = 1/(1 + exp(E/(kB * T))).

    kB in the above formula is Boltzmann constant.

    Parameters
    ----------

    energy :  float
        Energy values relative to Fermi level (meV)

    temperature : float
        Physical temperature in Kelvin

    Returns
    -------
    Occupation probability array [0,1]
    """
    return 1.0 / (1.0 + np.exp(energy / (_BOLTZMANN_CONSTANT * temperature)))
