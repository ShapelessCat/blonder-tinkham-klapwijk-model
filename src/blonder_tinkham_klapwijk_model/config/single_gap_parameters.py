from collections.abc import Mapping
from dataclasses import dataclass, asdict
from typing import final

from .wave_type import WaveType


@final
@dataclass(frozen=True)
class SingleGapParameters(Mapping):
    # Shared
    max_voltage: float  # mV
    n_points: int  # dimensionless
    d_temperature: float  # K
    temperature: float  # K
    angle: int  # degree
    # single gap specific
    proportion: float
    broadening_parameter: float  # Γ (meV)
    barrier_strength: float      # Z (dimensionless)
    gap: float  # Δ (meV)
    wave_type: WaveType

    def __getitem__(self, key):
        return asdict(self)[key]

    def __iter__(self):
        return iter(asdict(self))

    def __len__(self):
        return len(asdict(self))
