import logging
from collections.abc import Mapping
from typing import List, Self, final

from pydantic import BaseModel, model_validator

from .single_gap_parameters import SingleGapParameters
from .wave_type import WaveType


class _FrozenBasedModelWithMapping(BaseModel, Mapping):
    def __getitem__(self, key):
        return self.model_dump()[key]

    def __iter__(self):
        return iter(self.model_dump())

    def __len__(self):
        return len(self.model_fields())

    class Config:
        frozen = True  # Makes the model immutable


# Usage
@final
class SharedParameters(_FrozenBasedModelWithMapping):
    max_voltage: float  # mV
    n_points: int  # dimensionless
    d_temperature: float  # K
    temperature: float  # K
    angle: int  # degree


@final
class WaveSpecificParameters(_FrozenBasedModelWithMapping):
    proportion: float
    broadening_parameter: float  # Γ (meV)
    barrier_strength: float  # Z (dimensionless)
    gap: float  # Δ (meV)
    wave_type: WaveType


@final
class AppConfig(BaseModel):
    shared_parameters: SharedParameters
    wave_specific_parameters: List[WaveSpecificParameters]

    @model_validator(mode="after")
    def validate_voltages(self) -> Self:
        if (v := sum(wgp.proportion for wgp in self.wave_specific_parameters)) != 1:
            raise ValueError(f"The sum of all proportions should be 1. Now it's {v}")
        return self

    def config_set(self, index: int) -> SingleGapParameters:
        try:
            wave_specific_parameters = self.wave_specific_parameters[index]
        except IndexError:
            logging.error(
                f"Config only has {len(self.wave_specific_parameters)} set(s) of gap specific parameters."
            )
            raise

        return SingleGapParameters(**self.shared_parameters, **wave_specific_parameters)

    class Config:
        frozen = True
