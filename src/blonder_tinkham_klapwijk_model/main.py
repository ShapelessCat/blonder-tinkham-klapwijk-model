import numpy as np
from numpy.typing import NDArray

from blonder_tinkham_klapwijk_model.btk_fit import plot_experiment_result, plot_btk_tunneling_fit
from blonder_tinkham_klapwijk_model.config import load_config


CONFIG_PATH = "/Users/shapeless_cat/Projects/blonder-tinkham-klapwijk-model/src/blonder_tinkham_klapwijk_model/input_parameters.toml"
DATA_PATH = "/Users/shapeless_cat/Projects/blonder-tinkham-klapwijk-model/src/blonder_tinkham_klapwijk_model/3K_modified.dat"

def main():
    # print("Hello from blonder-tinkham-klapwijk-model!\n")
    #
    # config = load_config(CONFIG_PATH)
    # print(f"Full config is: \n{repr(config)}", end='\n\n')
    #
    # for i in range(len(config.gap_specific_parameters)):
    #     print(config.config_set(i))

    app_config_ = load_config(CONFIG_PATH)

    data: NDArray[np.float64] = np.loadtxt(DATA_PATH)
    bias_voltage_: NDArray[np.float64] = data[:, 0]
    normalized_conductance_: NDArray[np.float64] = data[:, 1]
    plot_experiment_result(app_config_.shared_parameters.max_voltage, bias_voltage=bias_voltage_, normalized_conductance=normalized_conductance_)

    plot_btk_tunneling_fit(app_config_)

if __name__ == '__main__':
    main()
