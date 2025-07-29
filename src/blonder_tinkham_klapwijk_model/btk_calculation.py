import operator
from dataclasses import dataclass
from functools import reduce
from typing import Self, final

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.interpolate import make_interp_spline

from . import ABS_ERR_TOLERANCE, HALF_PI
from .config import AppConfig
from .config.gap_config import ComplexGapConfig, SimpleGapConfig
from .config.wave_type import WaveType
from .fermi_window_for_tunneling import fermi_window_for_tunneling
from .transparency import normal_transparency_of
from .waves.anisotropic_wave import anisotropic_wave
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


def plot_experiment_result(
    max_vol: float,
    *,
    bias_voltage: NDArray[np.float64],
    normalized_conductance: NDArray[np.float64],
):
    plt.figure(1)
    plt.plot(bias_voltage, normalized_conductance, 'ob', markersize=1)

    plt.xlabel('Bias Voltage (mV)')
    plt.ylabel('Normalized Conductance')

    plt.xlim((-max_vol, max_vol))
    plt.tick_params(axis='both', labelsize=6)
    plt.xticks(
        np.arange(start=-max_vol, stop=max_vol + max_vol / 10, step=max_vol / 10)
    )
    # TODO: yticks
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.pause(0.1)


def plot_btk_tunneling_fit(app_conf: AppConfig) -> None:
    """BTK fitting code.

    Parameters
    __________
    input_parameters : AppConfig
    """
    summarized_gap_characteristics = reduce(
        operator.add,
        (
            calculate_gap_characteristics(**app_conf.config_set(idx))
            for idx in range(len(app_conf.wave_specific_parameters))
        ),
    )
    voltage = summarized_gap_characteristics.voltage
    current = summarized_gap_characteristics.current
    normalized_conductance = summarized_gap_characteristics.normalized_conductance
    # --- Plot results ---
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(voltage, normalized_conductance, 'r-', linewidth=2)
    ax2.plot(voltage, current, 'b-', linewidth=2)

    ax1.set_xlabel('Bias Voltage - V (mV)')
    ax1.set_ylabel('Normalized Conductance - dI/dV (unit?)', color='r')
    ax2.set_ylabel('Current - I (mA)', color='b')

    max_voltage = app_conf.shared_parameters.max_voltage
    ax1.set_xlim((-max_voltage, max_voltage))
    # ax1.set_ylim((0.99, 1.02))
    # ax2.set_ylim(-max_voltage, max_voltage)
    plt.show()

    # --- Save results ---
    Z: NDArray[np.float64] = np.column_stack((voltage, normalized_conductance, current))
    # parameters: NDArray[np.float64] = np.array([
    #     [app_conf.shared_parameters.temperature, proportion, 22],
    #     [gamma1, barrier1, gap1],
    #     [gamma2, barrier2, gap2],
    #     [angle1, angle2, 0]
    # ])
    # np.savetxt(path2result, Z, fmt='%.6f')
    #
    # print(f"Computation time: {time.time() - start_time:.2f} seconds")


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
    gap_config: SimpleGapConfig | ComplexGapConfig,  # Δ (meV)
    wave_type: WaveType,
) -> GapCharacteristics:
    match (wave_type, gap_config):
        case (WaveType.ANISOTROPIC, _):
            return calculate_anisomorphic_gap_characteristics(
                max_voltage,
                n_points,
                d_temperature,
                temperature,
                angle,
                proportion,
                broadening_parameter,
                barrier_strength,
                gap_config,
            )
        case (WaveType.ISOTROPIC, _) if isinstance(gap_config, SimpleGapConfig):
            return calculate_isomorphic_gap_characteristics(
                max_voltage,
                n_points,
                d_temperature,
                temperature,
                proportion,
                broadening_parameter,
                barrier_strength,
                gap_config.gap,
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
    gap_config: SimpleGapConfig | ComplexGapConfig,  # Δ (meV)
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
        return anisotropic_wave(
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


def calculate_isomorphic_gap_characteristics(
    # shared
    max_voltage: float,  # mV
    n_points: int,  # dimensionless
    d_temperature: float,  # K
    temperature: float,  # K
    # single gap specific
    proportion: float,
    broadening_parameter: float,  # Γ (meV)
    barrier_strength: float,  # Z (dimensionless)
    gap: float,  # Δ (meV)
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
            e, broadening_parameter, barrier_strength, gap, normalization_conductance_factor
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
