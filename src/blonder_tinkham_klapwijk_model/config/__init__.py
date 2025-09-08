import tomllib

from .app_config import (
    AppConfig,
    SharedParameters,
    SingleGapParameters,
    AtomicOrbitalSpecificParameters,
)

__all__ = [
    "AppConfig",
    "SharedParameters",
    "AtomicOrbitalSpecificParameters",
    "SingleGapParameters",
    "load_config",
]


def load_config(file_path: str) -> AppConfig:
    config_data = tomllib.load(open(file_path, "rb"))
    return AppConfig(**config_data)
