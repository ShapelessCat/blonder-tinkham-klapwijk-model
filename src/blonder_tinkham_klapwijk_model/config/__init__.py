import tomllib

from .app_config import (
    AppConfig,
    SharedParameters,
    SingleGapParameters,
    WaveSpecificParameters,
)

__all__ = [
    "AppConfig",
    "SharedParameters",
    "WaveSpecificParameters",
    "SingleGapParameters",
    "load_config",
]


def load_config(file_path: str) -> AppConfig:
    config_data = tomllib.load(open(file_path, "rb"))
    return AppConfig(**config_data)
