from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import final

from .gap_config import IsotropicGapConfig, AnisotropicGapConfig
from .atomic_orbital import AtomicOrbital


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
    barrier_strength: float  # Z (dimensionless)
    gap_config: IsotropicGapConfig | AnisotropicGapConfig  # Δ (meV)
    atomic_orbital: AtomicOrbital

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter({field.name: getattr(self, field.name) for field in fields(self)})

    def __len__(self):
        return len(fields(self))
