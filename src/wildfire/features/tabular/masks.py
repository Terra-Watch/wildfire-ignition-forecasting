"""Keep missing data missing: NaN propagation and consistency checks."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mask_dependent_columns(df: pd.DataFrame, key: str, cols: list[str]) -> int:
    """Where `key` is NaN, set every column in `cols` to NaN too (in place).

    The fire-weather recursion and the running rain and dry-spell counters return their initial state
    for grid cells whose weather cell is masked (sea or lake); those look like valid numbers. Returns the rows masked.
    """
    bad = df[key].isna().values
    for c in cols:
        if c in df:
            df.loc[bad, c] = np.nan
    return int(bad.sum())


def assert_same_nan_rows(df: pd.DataFrame, cols: list[str]) -> int:
    """Every column in `cols` must be NaN on exactly the same rows. Returns the number of NaN rows."""
    m = df[list(cols)].isna().values
    assert (m == m[:, :1]).all(), "columns disagree on which rows are NaN: " + str({c: int(df[c].isna().sum()) for c in cols})
    return int(m[:, 0].sum())


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Percentile rank in (0, 1] across cells, average ties."""
    return pd.Series(values).rank(pct=True, method="average").values.astype(np.float32)


def common_row_mask(df: pd.DataFrame, sets: dict, arms=None) -> np.ndarray:
    """Rows on which every listed arm has all its input columns non-NaN."""
    cols = sorted({c for a in (arms or sets.keys()) for c in sets[a]})
    return df[cols].notna().all(axis=1).values
