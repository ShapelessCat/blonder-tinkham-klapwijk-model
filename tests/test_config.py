from blonder_tinkham_klapwijk_model.config import AppConfig, load_config
from tests import test_data_home

CONFIG_PATH = test_data_home / "input_parameters.toml"


def test_config_loading():
    app_config_ = load_config(CONFIG_PATH)
    assert isinstance(app_config_, AppConfig)
