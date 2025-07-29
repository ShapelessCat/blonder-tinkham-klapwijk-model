import logging
from collections.abc import Iterator
from typing import Any, List, Mapping, Self, final

from pydantic import BaseModel, model_validator

from .gap_config import ComplexGapConfig, SimpleGapConfig
from .single_gap_parameters import SingleGapParameters
from .wave_type import WaveType


class FrozenBaseModel(BaseModel):
    class Config:
        frozen = True  # Makes the model immutable


# Usage
@final
class SharedParameters(FrozenBaseModel):
    max_voltage: float  # mV
    n_points: int  # dimensionless
    d_temperature: float  # K
    temperature: float  # K
    angle: int  # degree


@final
class WaveSpecificParameters(FrozenBaseModel, Mapping):
    proportion: float
    broadening_parameter: float  # Γ (meV)
    barrier_strength: float  # Z (dimensionless)
    gap_config: SimpleGapConfig | ComplexGapConfig  # Δ (meV)
    wave_type: WaveType

    @model_validator(mode='after')
    def check_field_conflict(self):
        match self.wave_type:
            case WaveType.ISOTROPIC if isinstance(
                gc := self.gap_config, ComplexGapConfig
            ):
                raise ValueError(f"Isotropic wave can't have complex gap config: {gc}")
            case _:
                pass
        return self

    def keys(self) -> Iterator[str]:
        return iter(self.__class__.model_fields.keys())

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        for name in self.__class__.model_fields.keys():
            yield name, getattr(self, name)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __len__(self):
        return len(self.mod)


@final
class AppConfig(FrozenBaseModel):
    shared_parameters: SharedParameters
    wave_specific_parameters: List[WaveSpecificParameters]

    @model_validator(mode="after")
    def validate_voltages(self) -> Self:
        if (v := sum(wgp.proportion for wgp in self.wave_specific_parameters)) != 1:
            raise ValueError(f"The sum of all proportions should be 1. Now it's {v}")
        return self

    def config_set(self, index: int) -> SingleGapParameters:
        try:
            wsp = self.wave_specific_parameters[index]
        except IndexError:
            logging.error(
                f"Config only has {len(self.wave_specific_parameters)} set(s) of gap specific parameters."
            )
            raise
        return SingleGapParameters(**self.shared_parameters.model_dump(), **wsp)
