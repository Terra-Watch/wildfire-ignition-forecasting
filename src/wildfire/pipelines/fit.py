"""Fitting helpers shared by the scoring pipelines."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.models.fit import Model, thin


def arm_rows(table: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Rows on which every input of the arm is present."""
    return table[cols].notna().all(axis=1).values if cols else np.ones(len(table), bool)


def arm_spec(arm: str) -> dict:
    return config.load("models")["arms"][arm]


def final_fit(table: pd.DataFrame, arm: str, columns: dict, years_ok, cap: int | None = None, thin_seed: int = 1) -> Model:
    """Fit one arm on the rows allowed by `years_ok`, optionally thinned to `cap` rows (all onsets kept)."""
    spec = arm_spec(arm)
    cols = columns[arm]
    m = arm_rows(table, cols) & years_ok
    x, y = table[cols].values[m], table.label.values[m]
    if cap:
        k = thin(y, cap, seed=thin_seed)
        x, y = x[k], y[k]
    return Model(spec["kind"], spec["params"]).fit(x, y)


def forward_folds(first_year: int | None = None, years: list[int] | None = None):
    """(validation year, fit years): each fold fits on every earlier year from the first year."""
    cfg = config.load("evaluation")["splits"]
    first = cfg["first_year"] if first_year is None else first_year
    return [(y, list(range(first, y))) for y in (cfg["forward_fold_years"] if years is None else years)]
