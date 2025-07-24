import numpy as np


def normal_transparency_of(cos_theta: float, barrier_strength: float) -> float:
    if np.isclose(barrier_strength, 0.0):
        return 1.0
    else:
        return cos_theta**2 / (cos_theta**2 + barrier_strength**2)


def superconductor_transparency_of(
    normal_transparency: float, gamma_plus: complex, gamma_minus: complex
) -> complex:
    numerator = (
        1
        + normal_transparency * np.abs(gamma_plus) ** 2
        + (normal_transparency - 1) * np.abs(gamma_plus * gamma_minus) ** 2
    )

    # Handle edge cases where gamma values are zero or very small
    if np.isclose(gamma_plus, 0.0) or np.isclose(gamma_minus, 0.0):
        return numerator

    # Safely compute the phase term
    try:
        ratio = (gamma_plus * np.abs(gamma_minus)) / (gamma_minus * np.abs(gamma_plus))
        # Ensure we don't take log of zero or negative real numbers
        if np.isclose(ratio, 0.0):
            return numerator
        phase = -1j * np.log(ratio)
        denominator = (
            np.abs(
                1
                + (normal_transparency - 1)
                * gamma_plus
                * gamma_minus
                * np.exp(1j * phase)
            )
            ** 2
        )
        return numerator / denominator
    except (ZeroDivisionError, RuntimeWarning):
        return numerator
