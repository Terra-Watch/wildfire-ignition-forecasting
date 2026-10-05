"""Reading the final settings (configs/final.yaml)."""
from __future__ import annotations

from pathlib import Path

import yaml

from deep_multimodal.model import ModelSettings
from deep_multimodal.training import TrainingSettings

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "final.yaml"


def load_settings(path: Path | str = DEFAULT_CONFIG) -> tuple[ModelSettings, TrainingSettings, dict]:
    """Returns (model settings, training settings, the whole configuration dictionary)."""
    config = yaml.safe_load(Path(path).read_text())
    training = dict(config["training"])
    training["seeds"] = tuple(training["seeds"])
    return ModelSettings(**config["model"]), TrainingSettings(**training), config
