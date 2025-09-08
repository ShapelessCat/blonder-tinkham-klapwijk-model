from dataclasses import dataclass
from typing import final


@final
@dataclass(frozen=True)
class IsotropicGapConfig:
    gap: float


@final
@dataclass(frozen=True)
class AnisotropicGapConfig:
    gap_plus: float
    gap_minus: float
