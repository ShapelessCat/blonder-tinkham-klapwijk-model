from enum import StrEnum, auto


class WaveType(StrEnum):
    ISOTROPIC = auto()
    ANISOTROPIC = auto()


class AtomicOrbitalType(StrEnum):
    S = auto()
    D = auto()
    # We don't need other atomic orbital types for now
