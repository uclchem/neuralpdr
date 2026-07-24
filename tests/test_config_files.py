from pathlib import Path

import pytest

from neuralpdr.config import read_conf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"

PATH_FIELDS = [
    "dataset_path",
    "input_features_file",
    "save_file_path",
    "normalisations_file",
]


def _discover_configs() -> list[Path]:
    """Every base/experiment TOML config that ships in the repo.

    Complements tests/test_config.py's targeted unit tests (Split,
    LearningScheme) by exercising every real config file on disk end to end.
    Excludes configs/*/features/**, which hold Features/Norms schemas rather
    than the Latent/FNO `Conf` schema `read_conf` validates against.
    """
    return sorted(
        p
        for p in CONFIGS_DIR.rglob("*.toml")
        if "features" not in p.relative_to(CONFIGS_DIR).parts
    )


ALL_CONFIGS = _discover_configs()
CONFIG_IDS = [str(p.relative_to(CONFIGS_DIR)) for p in ALL_CONFIGS]


@pytest.mark.parametrize("config_path", ALL_CONFIGS, ids=CONFIG_IDS)
def test_config_loads(config_path):
    config = read_conf(config_path)
    assert config.enc_dec_depth > 0
    assert config.batch_size > 0


@pytest.mark.parametrize("config_path", ALL_CONFIGS, ids=CONFIG_IDS)
def test_config_paths_are_absolute(config_path):
    config = read_conf(config_path)
    for field in PATH_FIELDS:
        value = getattr(config, field, None)
        if value:
            assert Path(value).is_absolute(), f"{field} is not absolute: {value}"
