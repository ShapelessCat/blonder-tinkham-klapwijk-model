import numpy as np


def gamma_function_of(energy: complex, delta: float) -> complex:
    return energy / np.abs(delta) - np.sqrt((energy / np.abs(delta)) ** 2 - 1)
