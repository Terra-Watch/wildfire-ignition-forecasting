"""Validity checks on built labels and features (pure functions; raise or return counts)."""
from __future__ import annotations

import numpy as np

from wildfire.labels.negatives import KEY_STRIDE, excluded_keys


def negatives_clear_of_detections(rows, cd) -> int:
    """Negatives that fall on a cell-day within the exclusion window after a detection. Must be 0."""
    neg = rows[rows.kind == "negative"]
    keys = neg.cell.values.astype(np.int64) * KEY_STRIDE + neg.day.values
    return int(np.isin(keys, excluded_keys(cd)).sum())


def training_rows_outside_train_role(rows, cells) -> int:
    """Rows of the training group that sit in a non-training cell. Must be 0."""
    t = rows[rows.group == "train"]
    return int((cells.role.values[t.cell.values] != "train").sum())


def climatology_first_year_is_nan(rows, column: str, first_year: int) -> int:
    """Non-NaN climatology values in the first year (no earlier year exists). Must be 0."""
    r = rows[rows.year == first_year]
    return int(r[column].notna().sum())


def burn_count_brute_force(row_day, neighbour_burns, lag: int, lookback: int) -> float:
    """log1p of burns dated in [day - lookback, day - lag) among `neighbour_burns` (a flat array of burn dates)."""
    v = np.asarray(neighbour_burns)
    sel = (v > 0) & (v < row_day - lag) & (v >= row_day - lookback)
    return float(np.log1p(sel.sum()))
