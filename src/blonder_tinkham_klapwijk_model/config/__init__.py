import tomllib

from .app_config import (
    AppConfig,
    GapSpecificParameters,
    SharedParameters,
    SingleGapParameters,
)

__all__ = [
    "AppConfig",
    "SharedParameters",
    "GapSpecificParameters",
    "SingleGapParameters",
    "load_config",
]


def load_config(file_path: str) -> AppConfig:
    config_data = tomllib.load(open(file_path, "rb"))
    return AppConfig(**config_data)
