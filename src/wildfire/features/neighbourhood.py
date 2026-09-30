"""Neighbourhood features: summaries of the cells around each cell, observed before the target day.

Rings are square (Chebyshev radius r, centre excluded). Positions with no study cell are not available, so edge
cells use the neighbours that exist. Onset counts use days D-28 .. D-1, so they never include the target day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import convolve

from wildfire import config
from wildfire.features import weather_windows
from wildfire.features.base import TIER_SUMMARY, FeatureSet, register
from wildfire.features.tabular import burn as burn_mod
from wildfire.features.tabular import weather
from wildfire.labels.detections import day_number, load_grid

VARIABLES = ["tree_cover", "burn_count", "years_since_burn", "vegetation_index", "fire_weather_index", "onsets_recent"]
SUM_VARIABLES = ("onsets_recent",)


def settings() -> dict:
    return config.load("baseline")["features"]["neighbourhood"]


def column_names() -> list[str]:
    return [f"{ring}_{v}" for ring in settings()["rings"] for v in VARIABLES]


def ring_stat(raster: np.ndarray, radius: int, total: bool = False) -> np.ndarray:
    """Mean (or sum) over the available ring cells at every position; NaN where none has a value."""
    k = np.ones((2 * radius + 1, 2 * radius + 1))
    k[radius, radius] = 0
    ok = ~np.isnan(raster)
    s = convolve(np.where(ok, raster, 0.0), k, mode="constant", cval=0.0)
    n = convolve(ok.astype(float), k, mode="constant", cval=0.0)
    return np.where(n > 0, s if total else s / np.maximum(n, 1), np.nan)


def to_raster(values: np.ndarray, cells: pd.DataFrame, shape: tuple) -> np.ndarray:
    """Place per-cell values on the (nx, ny) grid; positions with no study cell are NaN."""
    r = np.full(shape, np.nan)
    r[cells.ix.values, cells.iy.values] = values
    return r


def burn_by_day(days: np.ndarray, ncell: int) -> tuple[np.ndarray, np.ndarray]:
    """Per target day and cell: log1p of burns in the look-back window (NaN if too little was observed) and the capped
    years since the last observed burn. Shapes (days, cells)."""
    cfg = config.load("features")["tabular"]["burn"]
    b = burn_mod.load_burn(config.data_file("burn"))
    count = np.full((len(days), ncell), np.nan, np.float32)
    years = np.full((len(days), ncell), np.nan, np.float32)
    patterns, inverse = np.unique(b["covered"].T, axis=0, return_inverse=True)
    lo, hi = days - cfg["lookback_days"], days - cfg["lag_days"]
    obs = [burn_mod.window_coverage(p.astype(float), b["month_start"], b["month_end"], lo, hi) for p in patterns]
    for c in range(ncell):
        dates = np.sort(b["burn_ord"][:, c][b["burn_ord"][:, c] > 0]).astype(np.int64)
        n = np.searchsorted(dates, hi, "left") - np.searchsorted(dates, lo, "left")
        last = np.searchsorted(dates, hi, "left") - 1
        ref = dates[np.clip(last, 0, len(dates) - 1)] if len(dates) else np.zeros(len(days), np.int64)
        ref = np.where(last >= 0, ref, cfg["archive_start_day"])
        years[:, c] = np.minimum((days - ref) / 365.25, cfg["years_since_cap"])
        count[:, c] = np.where(obs[inverse[c]] >= cfg["min_observed_share"] - 1e-9, np.log1p(n), np.nan)
    return count, years


def vegetation_by_day(days: np.ndarray) -> np.ndarray:
    """Vegetation index of the latest composite ending before day - 1, NaN where the composite saw too little."""
    z = np.load(config.data_root() / "processed" / "ndvi_cells.npz")
    order = np.argsort(z["end"])
    nd, end = z["ndvi"][order], z["end"][order]
    k = np.searchsorted(end, days - 1, "left") - 1
    coverage = (~np.isnan(nd)).mean(1)
    out = nd[np.clip(k, 0, None)].copy()
    bad = (k < 0) | (coverage[np.clip(k, 0, None)] < settings()["vegetation_min_composite_coverage"])
    out[bad] = np.nan
    return out


def recent_onset_counts(onsets: pd.DataFrame, ncell: int, days: np.ndarray) -> np.ndarray:
    """Onsets in the 28 days before each target day, per cell: (days, cells)."""
    n_days = config.load("baseline")["features"]["neighbourhood"]["recent_onset_count_days"]
    first = int(onsets.day.min()) - 1
    grid = np.zeros((int(onsets.day.max()) - first + 2, ncell), np.int16)
    np.add.at(grid, (onsets.day.values - first, onsets.cell.values), 1)
    cum = np.cumsum(grid, axis=0)
    hi = np.clip(days - 1 - first, 0, len(cum) - 1)
    lo = np.clip(days - 1 - n_days - first, 0, len(cum) - 1)
    return (cum[hi] - cum[lo]).astype(np.float32)


def fire_weather_by_day(series: dict, days: np.ndarray) -> np.ndarray:
    """Fire-weather index of the day before each target day at every cell: (days, cells)."""
    t = days - 1 - series["first_day"]
    fwi = np.asarray(series["values"])[t][:, weather.NAMES.index("fwi"), :]
    return fwi[:, np.asarray(series["cell_point"])]


def _ring_columns(layers: dict, cells: pd.DataFrame, shape: tuple, tree: dict, rows: np.ndarray) -> np.ndarray:
    """The ring summaries of one target day at the given study cells: (len(rows), columns)."""
    out = np.empty((len(rows), len(column_names())), np.float32)
    ix, iy = cells.ix.values[rows], cells.iy.values[rows]
    j = 0
    for ring, radius in settings()["rings"].items():
        for v in VARIABLES:
            if v == "tree_cover":
                raster = tree[ring]
            else:
                raster = ring_stat(to_raster(layers[v], cells, shape), radius, total=v in SUM_VARIABLES)
            out[:, j] = raster[ix, iy]
            j += 1
    return out


def _tree_rings(cells: pd.DataFrame, shape: tuple) -> dict:
    tree = pd.read_csv(config.data_file("land_cover"), usecols=["wc_tree"]).wc_tree.values
    raster = to_raster(tree, cells, shape)
    return {ring: ring_stat(raster, radius) for ring, radius in settings()["rings"].items()}


def neighbourhood_matrix(cell: np.ndarray, day: np.ndarray, series: dict, onsets: pd.DataFrame) -> np.ndarray:
    """Neighbourhood features for (cell, target day) pairs: (rows, columns) float32."""
    cells, meta = load_grid()
    shape = (meta["nx"], meta["ny"])
    tree = _tree_rings(cells, shape)
    out = np.full((len(cell), len(column_names())), np.nan, np.float32)
    unique = np.unique(day)
    step = settings()["day_chunk"]
    for start in range(0, len(unique), step):
        days = unique[start:start + step]
        count, years = burn_by_day(days, len(cells))
        base = dict(burn_count=count, years_since_burn=years, vegetation_index=vegetation_by_day(days),
                    fire_weather_index=fire_weather_by_day(series, days),
                    onsets_recent=recent_onset_counts(onsets, len(cells), days))
        for k, d in enumerate(days):
            rows = np.flatnonzero(day == d)
            layers = {v: a[k] for v, a in base.items()}
            out[rows] = _ring_columns(layers, cells, shape, tree, cell[rows])
    return out


@register
class Neighbourhood(FeatureSet):
    name = "neighbourhood"
    tier = TIER_SUMMARY

    def columns(self) -> list[str]:
        return column_names()

    def build(self, keys):
        """Ring summaries for (cell, day) keys, from the stored weather series and the onsets table."""
        series = weather_windows.load_series(config.output_dir("features") / "daily_weather")
        onsets = pd.read_csv(config.output_dir("labels") / "cell_days.csv")
        onsets = onsets[onsets.onset][["cell", "day"]]
        m = neighbourhood_matrix(keys.cell.values, keys.day.values, series, onsets)
        return pd.DataFrame(m, columns=column_names(), index=keys.index)
