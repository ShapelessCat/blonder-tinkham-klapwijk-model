import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .btk_calculation import GapCharacteristics


def plot_experiment_measurement(
    max_vol: float,
    *,
    bias_voltage: NDArray[np.float64],
    normalized_conductance: NDArray[np.float64],
    fig_num: int = 1,
):
    """Plot experimental measurement data."""
    plt.figure(fig_num)
    plt.plot(
        bias_voltage,
        normalized_conductance,
        'ob',
        markersize=1,
        label='Experimental Data',
    )

    plt.xlabel('Bias Voltage (mV)')
    plt.ylabel('Normalized Conductance')

    plt.xlim((-max_vol, max_vol))
    plt.tick_params(axis='both', labelsize=6)
    plt.xticks(
        np.arange(start=-max_vol, stop=max_vol + max_vol / 10, step=max_vol / 10)
    )
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # Modified legend with smaller font and frame
    plt.legend(
        loc='upper left',
        fontsize=6,  # Smaller font size
        frameon=True,  # Keep frame visible
        framealpha=0.8,  # Slightly transparent
        handlelength=1.5,  # Shorter line segments in legend
        borderpad=0.3,  # Less padding inside frame
        borderaxespad=0.3,  # Less padding between axes and legend
        handletextpad=0.3,  # Less padding between symbol and text
    )
    plt.pause(0.1)


def plot_btk_tunneling_fit(
    summarized_gap_characteristics: GapCharacteristics, fig_num: int = 1
) -> None:
    """Plot BTK fitting results on existing figure."""
    plt.figure(fig_num)
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(
        summarized_gap_characteristics.voltage,
        summarized_gap_characteristics.normalized_conductance,
        'r-',
        linewidth=2,
        label='BTK Fit',
    )

    ax2.plot(
        summarized_gap_characteristics.voltage,
        summarized_gap_characteristics.current,
        'b-',
        linewidth=2,
        label='Current',
    )

    ax1.set_ylabel('Normalized Conductance - dI/dV', color='k')
    ax2.set_ylabel('Current - I (mA)', color='b')

    # Modified combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='upper left',
        fontsize=6,  # Smaller font size
        frameon=True,  # Keep frame visible
        framealpha=0.8,  # Slightly transparent
        handlelength=1.5,  # Shorter line segments
        borderpad=0.3,  # Less internal padding
        borderaxespad=0.3,  # Less padding to axes
        handletextpad=0.3,  # Less symbol-text padding
    )

    # Make the legend frame smaller
    frame = legend.get_frame()
    frame.set_linewidth(0.5)  # Thinner border line

    plt.tight_layout()
    plt.show()
