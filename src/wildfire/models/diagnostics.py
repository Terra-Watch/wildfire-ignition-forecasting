"""Diagnostic arms that measure the "which cell" shortcut: a model that knows only where the cell is."""
from __future__ import annotations

import numpy as np
import pandas as pd


def coordinate_matrix(cells: pd.DataFrame, cell_index: np.ndarray) -> np.ndarray:
    """Projected centroid (x, y) of each cell: (rows, 2) float32."""
    return cells[["cx", "cy"]].values[cell_index].astype(np.float32)


def cell_rates(fit_cell: np.ndarray, fit_label: np.ndarray, eval_cell: np.ndarray) -> np.ndarray:
    """Share of fit rows that are onsets, per cell, looked up for `eval_cell`; unseen cells get the overall share."""
    df = pd.DataFrame(dict(cell=fit_cell, y=fit_label))
    rate = df.groupby("cell").y.mean()
    return pd.Series(eval_cell).map(rate).fillna(float(np.mean(fit_label))).values
