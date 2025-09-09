from dataclasses import dataclass
from typing import final


@final
@dataclass(frozen=True)
class S0Formula:
    pass


@final
@dataclass(frozen=True)
class S1Formula:
    epsilon: float
    symmetry: int
