"""Arm definitions: which columns each arm reads."""
from __future__ import annotations

import pandas as pd

from wildfire import config


def feature_groups(table: pd.DataFrame | None = None) -> dict[str, list[str]]:
    """Feature groups from configs/features.yaml, with `static` assembled from its parts and `human` restricted to
    layers that are present and non-missing for at least the configured share of rows."""
    t = config.load("features")["tabular"]
    groups = {k: list(v) for k, v in t["groups"].items()}
    groups["static"] = [c for part in t["static_order"] for c in groups[part]]
    if table is not None:
        share = t["min_non_missing_share"]
        groups["human"] = [c for c in groups["human"] if c in table and table[c].notna().mean() >= share]
    return groups


def arm_columns(table: pd.DataFrame | None = None) -> dict[str, list[str]]:
    """{arm: input columns} for every arm that is fitted; the climatology arm reads its own column."""
    groups = feature_groups(table)
    arms = config.load("models")["arms"]
    clim = config.load("features")["tabular"]["climatology_column"]
    out = {}
    for name, spec in arms.items():
        out[name] = [clim] if spec["kind"] == "climatology" else [c for g in spec["inputs"] for c in groups[g]]
    return out


def fitted_arms() -> list[str]:
    return [a for a, s in config.load("models")["arms"].items() if s["kind"] != "climatology"]


def reference_arm() -> str:
    return next(a for a, s in config.load("models")["arms"].items() if s["kind"] == "climatology")
