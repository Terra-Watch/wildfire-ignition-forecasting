"""Day-ahead forecast inputs read from the saved forecast lookup (one record per forecast grid pixel and target day)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from deep_multimodal import constants as c

SCALES = {"t2m": 100, "rh": 10, "u10": 100, "v10": 100, "precip": 100}      # the lookup stores integers: value times scale
COLUMN_NAMES = [f"{v}_{s}" for v in c.FORECAST_VARIABLES for s in c.FORECAST_STATISTICS] + [f"{v}_h{h:02d}" for v in c.FORECAST_VARIABLES for h in c.FORECAST_HOURS]
COLUMN_SCALES = np.array([float(SCALES[name.split("_")[0]]) for name in COLUMN_NAMES], np.float32)
MISSING_CODE = -32768


def forecast_inputs(keys: np.ndarray, lookup_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For rows keyed by (cell, day) return hourly values (rows, 12, 5), daily statistics (rows, 20) and a flag that is True where the day has no forecast.
    Missing values are NaN; they are masked later, never filled with zeros."""
    lookup = Path(lookup_dir)
    table = np.load(lookup / "gfs_i16.npy", mmap_mode="r")
    days, cell_pixel = np.load(lookup / "days.npy"), np.load(lookup / "cell_pixel.npy")
    cell, day = np.asarray(keys[:, 0]), np.asarray(keys[:, 1])
    position = np.searchsorted(days, day)
    present = (position < len(days)) & (days[np.minimum(position, len(days) - 1)] == day)
    raw = np.asarray(table[cell_pixel[cell], np.minimum(position, len(days) - 1)])
    missing = (raw[:, 0] == MISSING_CODE) | ~present
    values = np.where((raw == MISSING_CODE) | ~present[:, None], np.nan, raw.astype(np.float32) / COLUMN_SCALES[None]).astype(np.float32)
    column = {name: i for i, name in enumerate(COLUMN_NAMES)}
    hourly = np.stack([[values[:, column[f"{v}_h{h:02d}"]] for v in c.FORECAST_VARIABLES] for h in c.FORECAST_HOURS])      # (12, 5, rows)
    hourly = hourly.transpose(2, 0, 1)
    daily = np.stack([values[:, column[f"{v}_{s}"]] for v in c.FORECAST_VARIABLES for s in c.FORECAST_STATISTICS], 1)
    return hourly.astype(np.float32), daily.astype(np.float32), missing
