import logging
from collections.abc import Mapping
from typing import Any, Generator, KeysView, final

from pydantic import BaseModel, model_validator, Field

from .formula_type import S0Formula, S1Formula
from .gap_config import AnisotropicGapConfig, IsotropicGapConfig
from .single_gap_parameters import SingleGapParameters
from .atomic_orbital import AtomicOrbital


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
class AtomicOrbitalSpecificParameters(FrozenBaseModel, Mapping):
    proportion: float
    broadening_parameter: float  # Γ (meV)
    barrier_strength: float  # Z (dimensionless)
    gap_config: IsotropicGapConfig | AnisotropicGapConfig  # Δ (meV)
    atomic_orbital: AtomicOrbital
    formula_type: S0Formula | S1Formula = Field(default_factory=S0Formula)

    @model_validator(mode='after')
    def check_field_conflict(self):
        match self.atomic_orbital:
            case AtomicOrbital.S:
                pass
            case t if isinstance(gc := self.gap_config, IsotropicGapConfig):
                raise ValueError(f"{t} atomic orbital can be isotropic wave: {gc}")
            case _:
                pass
        return self

    def keys(self) -> KeysView[str]:
        return self.__class__.model_fields.keys()

    def __iter__(self) -> Generator[tuple[str, Any], None, None]:
        for name in self.__class__.model_fields.keys():
            yield name, getattr(self, name)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __len__(self):
        return len(self.mod)


@final
class AppConfig(FrozenBaseModel):
    shared_parameters: SharedParameters
    atomic_orbital_specific_parameters: list[AtomicOrbitalSpecificParameters]

    # If I use `Self` as the return type,
    # mypy report error "Cannot infer function type argument  [misc]"
    @model_validator(mode="after")
    def validate_voltages(self) -> "AppConfig":
        if (v := sum(wgp.proportion for wgp in self.atomic_orbital_specific_parameters)) != 1:
            raise ValueError(f"The sum of all proportions should be 1. Now it's {v}")
        return self

    def config_set(self, index: int) -> SingleGapParameters:
        try:
            wsp = self.atomic_orbital_specific_parameters[index]
        except IndexError:
            logging.error(
                f"Config only has {len(self.atomic_orbital_specific_parameters)} set(s) of gap specific parameters."
            )
            raise
        return SingleGapParameters(**self.shared_parameters.model_dump(), **wsp)
