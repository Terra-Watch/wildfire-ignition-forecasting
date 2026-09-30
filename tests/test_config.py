from pathlib import Path

import pytest

from wildfire import config


def test_data_root_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert config.data_root() == tmp_path


def test_missing_data_root_raises(monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    with pytest.raises(KeyError):
        config.data_root()


def test_artifacts_root_has_default(monkeypatch):
    monkeypatch.delenv("ARTIFACTS_ROOT", raising=False)
    assert config.artifacts_root() == Path("./artifacts")


@pytest.mark.parametrize("name", ["domain", "data", "features", "models", "evaluation"])
def test_every_config_loads(monkeypatch, tmp_path, name):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert isinstance(config.load(name), dict)
