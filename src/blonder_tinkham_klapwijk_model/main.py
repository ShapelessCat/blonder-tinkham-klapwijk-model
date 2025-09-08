import argparse
import operator
import sys
from argparse import ArgumentParser
from functools import reduce

import numpy as np
from numpy.typing import NDArray

from blonder_tinkham_klapwijk_model.btk_calculation import (
    calculate_gap_characteristics,
)
from blonder_tinkham_klapwijk_model.config import load_config
from blonder_tinkham_klapwijk_model.plotting import (
    plot_btk_tunneling_fit,
    plot_experiment_measurement,
)

EPILOG = """
Example config file (specified through -c/--config):\n
  ```toml
  [shared_parameters]
  max_voltage = 30
  n_points = 151
  d_temperature = 10
  temperature = 5
  angle = 45 # degree
  
  [[atomic_orbital_specific_parameters]]
  proportion = 0.5
  broadening_parameter = 0.01                  # Γ (meV)
  barrier_strength = 20                        # Z - dimensionless
  gap_config = { gap_plus = 4, gap_minus = 4 } # Δ (meV)
  atomic_orbital = "s"
  
  [[atomic_orbital_specific_parameters]]
  proportion = 0.5
  broadening_parameter = 0.01  # Γ (meV)
  barrier_strength = 20        # Z - dimensionless
  gap_config = { gap = 8 }     # Δ (meV)
  atomic_orbital = "s"
  ```
"""


def build_parser() -> ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='BTK data plotting and fitting',
        formatter_class=argparse.RawTextHelpFormatter,  # Preserves formatting
        description='Plot BTK data from experiment and find the best curve that can fit it.',
        epilog=EPILOG,
    )
    parser.add_argument(
        '-d',
        '--data',
        help='Specify a data file path. The first column should be bias voltage (mV), and second column should be normalized conductance.',
    )
    parser.add_argument(
        '-c', '--config', help='Specify theoretical curve configuration.'
    )
    return parser


def main() -> None:
    parser = build_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    print(args.config, args.data)

    app_config_ = load_config(args.config)

    data: NDArray[np.float64] = np.loadtxt(args.data)
    bias_voltage_: NDArray[np.float64] = data[:, 0]
    normalized_conductance_: NDArray[np.float64] = data[:, 1]
    plot_experiment_measurement(
        app_config_.shared_parameters.max_voltage,
        bias_voltage=bias_voltage_,
        normalized_conductance=normalized_conductance_,
    )

    summarized_gap_characteristics = reduce(
        operator.add,
        (
            calculate_gap_characteristics(**app_config_.config_set(idx))
            for idx in range(len(app_config_.atomic_orbital_specific_parameters))
        ),
    )
    plot_btk_tunneling_fit(summarized_gap_characteristics)


if __name__ == '__main__':
    main()
