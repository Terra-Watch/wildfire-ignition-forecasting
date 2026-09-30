"""The single evaluation harness. Every feature set and model goes through it."""
from __future__ import annotations

from typing import Sequence

from wildfire.features.base import FeatureSet


def evaluate(feature_sets: Sequence[FeatureSet], model, split: str = "validation") -> dict:
    """Score `model` on the combined columns of `feature_sets`.

    The held-out split is refused until the final evaluation is requested.
    """
    if split not in {"train", "validation"}:
        raise ValueError(f"split '{split}' is not available for development")
    raise NotImplementedError
