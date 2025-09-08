from dataclasses import dataclass
from typing import Self, final

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.interpolate import make_interp_spline

from . import ABS_ERR_TOLERANCE, HALF_PI
from .config.gap_config import IsotropicGapConfig, AnisotropicGapConfig
from .config.atomic_orbital import AtomicOrbital
from .fermi_window_for_tunneling import fermi_window_for_tunneling
from .transparency import normal_transparency_of
from .waves.anisotropic_s import anisotropic_s
from .waves.d import d
from .waves.isotropic_wave import isotropic_wave


@final
@dataclass(frozen=True)
class GapCharacteristics:
    current: NDArray[np.float64]
    normalized_conductance: NDArray[np.float64]
    voltage: NDArray[np.float64]

    def __add__(self, other: Self) -> Self:
        if not np.array_equal(self.voltage, other.voltage):
            raise ValueError(
                "Only GapCharacteristics's with matched voltages can be added together."
            )
        return GapCharacteristics(
            self.current + other.current,
            self.normalized_conductance + other.normalized_conductance,
            self.voltage,
        )


def calculate_gap_characteristics(
    # shared
    max_voltage: float,  # mV
    n_points: int,  # dimensionless
    d_temperature: float,  # K
    temperature: float,  # K
    angle: int,  # degree
    # single gap specific
    proportion: float,
    broadening_parameter: float,  # Γ (meV)
    barrier_strength: float,  # Z (dimensionless)
    gap_config: IsotropicGapConfig | AnisotropicGapConfig,  # Δ (meV)
    atomic_orbital: AtomicOrbital,
) -> GapCharacteristics:
    match (atomic_orbital, gap_config):
        case (AtomicOrbital.S, IsotropicGapConfig() as isotropic_gap_config):
            return calculate_isomorphic_s_gap_characteristics(
                max_voltage,
                n_points,
                d_temperature,
                temperature,
                proportion,
                broadening_parameter,
                barrier_strength,
                isotropic_gap_config,
            )
        case (_, AnisotropicGapConfig() as anisotropic_gap_config):
            return calculate_anisomorphic_gap_characteristics(
                max_voltage,
                n_points,
                d_temperature,
                temperature,
                angle,
                proportion,
                broadening_parameter,
                barrier_strength,
                anisotropic_gap_config,
                atomic_orbital,
            )
        case _:
            raise ValueError("Should never reach here!")


def calculate_anisomorphic_gap_characteristics(
    # shared
    max_voltage: float,  # mV
    n_points: int,  # dimensionless
    d_temperature: float,  # K
    temperature: float,  # K
    angle: int,  # degree
    # single gap specific
    proportion: float,
    broadening_parameter: float,  # Γ (meV)
    barrier_strength: float,  # Z (dimensionless)
    gap_config: AnisotropicGapConfig,  # Δ (meV)
    atomic_orbital: AtomicOrbital,
) -> GapCharacteristics:
    def compute_normalization_conductance_factor() -> float:
        def f(theta: float):
            cos_v = np.cos(theta)
            nt = normal_transparency_of(cos_v, barrier_strength)
            return nt * cos_v

        return quad(f, -HALF_PI, HALF_PI)[0]

    normalization_conductance_factor: float = compute_normalization_conductance_factor()

    # --- DOS Calculation ---
    pointnum: int = 501
    energy: NDArray[np.float64] = np.zeros((pointnum + 1) // 2)
    dos0: NDArray[np.float64] = np.zeros_like(energy)

    def anisotropic_wave_(theta, e):
        match atomic_orbital:
            case AtomicOrbital.S:
                return anisotropic_s(
                    theta,
                    e,
                    broadening_parameter,
                    barrier_strength,
                    gap_config,
                    angle,
                    normalization_conductance_factor,
                )
            case AtomicOrbital.D:
                return d(
                    theta,
                    e,
                    broadening_parameter,
                    barrier_strength,
                    gap_config,
                    angle,
                    normalization_conductance_factor,
                )

    for n in range((pointnum + 1) // 2):
        energy[n] = n * (max_voltage + d_temperature + 1) / 500
        dos0[n] = quad(
            anisotropic_wave_,
            -HALF_PI,
            HALF_PI,
            epsabs=ABS_ERR_TOLERANCE,
            args=(energy[n].item(),),
            limit=200,
            complex_func=True,
        )[0].real

    # Mirror for negative energies
    energy_full: NDArray[np.float64] = np.concatenate((-energy[:0:-1], energy))
    dos0_full: NDArray[np.float64] = np.concatenate((dos0[:0:-1], dos0))

    # --- Current Calculation ---
    integral_result: NDArray[np.float64] = np.zeros(2 * n_points - 1)

    bias_voltages: NDArray[np.float64] = np.linspace(
        -max_voltage, max_voltage, 2 * n_points - 1
    )
    for n in range(n_points - 1, 2 * n_points - 1):
        bias_voltage: float = bias_voltages[n].item()
        intenergy: NDArray[np.float64] = np.linspace(
            -d_temperature - 1, bias_voltage + d_temperature + 1, 1001
        )
        fermifunc: NDArray[np.float64] = np.fromiter(
            (
                fermi_window_for_tunneling(e, bias_voltage, temperature)
                for e in intenergy
            ),
            dtype=np.float64,
        )
        intdos0: NDArray[np.float64] = make_interp_spline(energy_full, dos0_full)(
            intenergy
        )
        integral_result[n] = np.trapezoid(intdos0 * fermifunc, intenergy)

    integral_result[: n_points - 1] = -integral_result[
        2 * n_points - 2 : n_points - 1 : -1
    ]

    voltage: NDArray[np.float64] = np.linspace(-max_voltage, max_voltage, 200)
    current: NDArray[np.float64] = make_interp_spline(bias_voltages, integral_result)(
        voltage
    )
    didv: NDArray[np.float64] = np.gradient(current, voltage[1] - voltage[0])

    return GapCharacteristics(current * proportion, didv * proportion, voltage)


def calculate_isomorphic_s_gap_characteristics(
    # shared
    max_voltage: float,  # mV
    n_points: int,  # dimensionless
    d_temperature: float,  # K
    temperature: float,  # K
    # single gap specific
    proportion: float,
    broadening_parameter: float,  # Γ (meV)
    barrier_strength: float,  # Z (dimensionless)
    gap_config: IsotropicGapConfig,  # Δ (meV)
) -> GapCharacteristics:
    def compute_normalization_conductance_factor() -> float:
        return 1 / (1 + barrier_strength**2)

    normalization_conductance_factor: float = compute_normalization_conductance_factor()

    # --- DOS Calculation ---
    pointnum: int = 501
    energy: NDArray[np.float64] = np.zeros((pointnum + 1) // 2)
    dos0: NDArray[np.float64] = np.zeros_like(energy)

    def isotropic_wave_(e):
        return isotropic_wave(
            e,
            broadening_parameter,
            barrier_strength,
            gap_config.gap,
            normalization_conductance_factor,
        )

    # energy: float,
    # broadening_parameter: float,
    # barrier_strength: float,
    # gap: float,
    for n in range((pointnum + 1) // 2):
        energy[n] = n * (max_voltage + d_temperature + 1) / 500
        dos0[n] = isotropic_wave_(energy[n])

    # Mirror for negative energies
    energy_full: NDArray[np.float64] = np.concatenate((-energy[:0:-1], energy))
    dos0_full: NDArray[np.float64] = np.concatenate((dos0[:0:-1], dos0))

    # --- Current Calculation ---
    integral_result: NDArray[np.float64] = np.zeros(2 * n_points - 1)

    bias_voltages: NDArray[np.float64] = np.linspace(
        -max_voltage, max_voltage, 2 * n_points - 1
    )
    for n in range(n_points - 1, 2 * n_points - 1):
        bias_voltage: float = bias_voltages[n].item()
        intenergy: NDArray[np.float64] = np.linspace(
            -d_temperature - 1, bias_voltage + d_temperature + 1, 1001
        )
        fermifunc: NDArray[np.float64] = np.fromiter(
            (
                fermi_window_for_tunneling(e, bias_voltage, temperature)
                for e in intenergy
            ),
            dtype=np.float64,
        )
        intdos0: NDArray[np.float64] = make_interp_spline(energy_full, dos0_full)(
            intenergy
        )
        integral_result[n] = np.trapezoid(intdos0 * fermifunc, intenergy)

    integral_result[: n_points - 1] = -integral_result[
        2 * n_points - 2 : n_points - 1 : -1
    ]

    voltage: NDArray[np.float64] = np.linspace(-max_voltage, max_voltage, 200)
    current: NDArray[np.float64] = make_interp_spline(bias_voltages, integral_result)(
        voltage
    )
    didv: NDArray[np.float64] = np.gradient(current, voltage[1] - voltage[0])

    return GapCharacteristics(current * proportion, didv * proportion, voltage)
