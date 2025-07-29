from dataclasses import dataclass
from typing import final


@final
@dataclass(frozen=True)
class SimpleGapConfig:
    gap: float


@final
@dataclass(frozen=True)
class ComplexGapConfig:
    gap_plus: float
    gap_minus: float
