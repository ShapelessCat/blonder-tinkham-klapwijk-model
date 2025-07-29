import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .btk_calculation import GapCharacteristics


def plot_experiment_measurement(
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
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.pause(0.1)


def plot_btk_tunneling_fit(
    summarized_gap_characteristics: GapCharacteristics, max_voltage: float
) -> None:
    """BTK fitting code.

    Parameters
    __________
    summarized_gap_characteristics : GapCharacteristics

    max_voltage : float
    """
    voltage = summarized_gap_characteristics.voltage
    current = summarized_gap_characteristics.current
    normalized_conductance = summarized_gap_characteristics.normalized_conductance

    # --- Plot results ---
    fig, axes1 = plt.subplots()
    axes2 = axes1.twinx()

    axes1.plot(voltage, normalized_conductance, 'r-', linewidth=2)
    axes2.plot(voltage, current, 'b-', linewidth=2)

    axes1.set_xlabel('Bias Voltage - V (mV)')
    axes1.set_ylabel('Normalized Conductance - dI/dV (unit?)', color='r')
    axes2.set_ylabel('Current - I (mA)', color='b')

    axes1.set_xlim((-max_voltage, max_voltage))
    axes1.tick_params(axis='both', labelsize=6)
    axes1.set_xticks(
        np.arange(start=-max_voltage, stop=max_voltage + max_voltage / 10, step=max_voltage / 10)
    )
    axes1.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # ax1.set_ylim((0.99, 1.02))
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
