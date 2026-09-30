"""Tabular feature set: the 42 per-cell-day inputs (weather, fire-weather codes, static, human, season, burn history)."""
from __future__ import annotations

from wildfire import config
from wildfire.features.base import TIER_SUMMARY, FeatureSet, register


@register
class Tabular(FeatureSet):
    name = "tabular"
    tier = TIER_SUMMARY

    def columns(self) -> list[str]:
        """Input columns in the order of the widest arm."""
        from wildfire.models.arms import arm_columns
        widest = max(arm_columns().values(), key=len)
        return list(widest)

    def build(self, keys):
        """Features for (cell, day) keys, read from the built feature table."""
        import pandas as pd
        table = pd.read_pickle(config.output_dir("features") / "tabular.pkl")
        return keys[["cell", "day"]].merge(table[["cell", "day"] + self.columns()], on=["cell", "day"], how="left")
