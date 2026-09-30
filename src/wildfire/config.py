"""Load configs/*.yaml and resolve data locations."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
_VAR = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            found = os.environ.get(m.group(1), m.group(2))
            if found is None:
                raise KeyError(f"environment variable {m.group(1)} is not set")
            return found
        return os.path.expanduser(_VAR.sub(sub, value))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load(name: str, expand: bool = True) -> dict:
    data = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text()) or {}
    return _expand(data) if expand else data


def data_root() -> Path:
    return Path(_expand(load("data", expand=False)["data_root"]))


def artifacts_root() -> Path:
    return Path(_expand(load("data", expand=False)["artifacts_root"]))
