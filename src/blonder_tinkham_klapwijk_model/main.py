import argparse
import operator
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

CONFIG_PATH = "/Users/shapeless_cat/Projects/blonder-tinkham-klapwijk-model/src/blonder_tinkham_klapwijk_model/input_parameters.toml"
DATA_PATH = "/Users/shapeless_cat/Projects/blonder-tinkham-klapwijk-model/src/blonder_tinkham_klapwijk_model/3K_modified.dat"


def main():
    parser = argparse.ArgumentParser(
        prog='BTK data plotting and fitting',
        description='Plot BTK data from experiment and find the best curve that can fit it.',
    )
    parser.add_argument('-d', '--data')
    parser.add_argument('-c', '--config')
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
            for idx in range(len(app_config_.wave_specific_parameters))
        ),
    )
    plot_btk_tunneling_fit(
        summarized_gap_characteristics, app_config_.shared_parameters.max_voltage
    )


if __name__ == '__main__':
    main()
